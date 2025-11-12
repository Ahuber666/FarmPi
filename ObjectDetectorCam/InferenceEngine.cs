using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ObjectDetectorCam
{
    public sealed class InferenceEngine : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;
        private readonly int _inW;
        private readonly int _inH;
        private readonly List<string> _labels;

        public InferenceEngine(string modelPath, string labelsPath, (int W, int H)? knownWH = null)
        {
            if (!File.Exists(modelPath)) throw new FileNotFoundException("ONNX model not found.", modelPath);
            if (!File.Exists(labelsPath)) throw new FileNotFoundException("labels.txt not found.", labelsPath);

            _labels = File.ReadAllLines(labelsPath)
                          .Where(l => !string.IsNullOrWhiteSpace(l))
                          .Select(l => l.Trim())
                          .ToList();

            // CPU by default. If you have CUDA installed, you can enable it:
            // var options = SessionOptions.MakeSessionOptionWithCudaProvider();
            var options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;

            _session = new InferenceSession(modelPath, options);

            // Choose first input by default
            _inputName = _session.InputMetadata.Keys.First();

            if (knownWH.HasValue)
            {
                _inW = knownWH.Value.W;
                _inH = knownWH.Value.H;
            }
            else
            {
                // Try to infer from input shape [1,3,H,W]
                var md = _session.InputMetadata[_inputName];
                var dims = md.Dimensions.Select(d => d > 0 ? d : 0).ToArray();
                // Fallback if dynamic:
                _inH = dims.Length >= 3 ? Math.Abs(dims[^2]) : 640;
                _inW = dims.Length >= 4 ? Math.Abs(dims[^1]) : 640;
                if (_inW == 0 || _inH == 0) { _inW = 640; _inH = 640; }
            }
        }

        public List<Detection> Detect(Mat bgrFrame, float scoreThreshold, float nmsThreshold)
        {
            // Preprocess
            using var resized = new Mat();
            Cv2.Resize(bgrFrame, resized, new OpenCvSharp.Size(_inW, _inH));
            Cv2.CvtColor(resized, resized, ColorConversionCodes.BGR2RGB);

            // Normalize to 0..1, CHW float32
            var tensor = new DenseTensor<float>(new[] { 1, 3, _inH, _inW });
            var idx = 0;
            unsafe
            {
                // Faster copy via Span
                for (int y = 0; y < _inH; y++)
                {
                    var row = resized.Row(y);
                    for (int x = 0; x < _inW; x++)
                    {
                        var vec = row.Get<Vec3b>(0, x);
                        tensor.Buffer.Span[idx + 0 * _inH * _inW] = vec.Item0 / 255f; // R
                        tensor.Buffer.Span[idx + 1 * _inH * _inW] = vec.Item1 / 255f; // G
                        tensor.Buffer.Span[idx + 2 * _inH * _inW] = vec.Item2 / 255f; // B
                        idx++;
                    }
                }
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, tensor)
            };

            using var results = _session.Run(inputs);

            // Try to support common object-detection ONNX heads:
            // A) Custom Vision style: boxes [1,N,4], scores [1,N], labels [1,N]
            // B) YOLO-style: one output [1,N,6] = [x1,y1,x2,y2,score,class]
            var outputNames = results.Select(r => r.Name).ToList();
            var dets = new List<Detection>();

            if (Has(outputs: outputNames, "boxes") && Has(outputs: outputNames, "scores") && Has(outputs: outputNames, "labels"))
            {
                var boxesTensor = results.First(r => r.Name.Contains("boxes")).AsEnumerable<float>().ToArray();
                var scoresTensor = results.First(r => r.Name.Contains("scores")).AsEnumerable<float>().ToArray();
                var labelsTensor = results.First(r => r.Name.Contains("labels")).AsEnumerable<long>().ToArray();

                // Determine N
                int n = scoresTensor.Length;

                // boxes layout assumptions:
                // If values look <= 1, treat as normalized [x, y, w, h] (Custom Vision).
                // Else treat as absolute [x1, y1, x2, y2] relative to model input size.
                bool normalized = boxesTensor.Length >= 4 && boxesTensor.Take(Math.Min(100, boxesTensor.Length)).Max() <= 1.5f;

                for (int i = 0; i < n; i++)
                {
                    float score = scoresTensor[i];
                    if (score < scoreThreshold) continue;

                    int bi = i * 4;
                    float x1, y1, x2, y2;

                    if (normalized)
                    {
                        float nx = boxesTensor[bi + 0];
                        float ny = boxesTensor[bi + 1];
                        float nw = boxesTensor[bi + 2];
                        float nh = boxesTensor[bi + 3];
                        x1 = nx * bgrFrame.Cols;
                        y1 = ny * bgrFrame.Rows;
                        x2 = (nx + nw) * bgrFrame.Cols;
                        y2 = (ny + nh) * bgrFrame.Rows;
                    }
                    else
                    {
                        // absolute in model space; map back to frame size
                        float ax1 = boxesTensor[bi + 0];
                        float ay1 = boxesTensor[bi + 1];
                        float ax2 = boxesTensor[bi + 2];
                        float ay2 = boxesTensor[bi + 3];

                        float sx = (float)bgrFrame.Cols / _inW;
                        float sy = (float)bgrFrame.Rows / _inH;

                        x1 = ax1 * sx;
                        y1 = ay1 * sy;
                        x2 = ax2 * sx;
                        y2 = ay2 * sy;
                    }

                    var rect = ClampToFrame(x1, y1, x2, y2, bgrFrame.Cols, bgrFrame.Rows);
                    int classId = (int)labelsTensor[i];
                    string label = (classId >= 0 && classId < _labels.Count) ? _labels[classId] : $"cls{classId}";
                    dets.Add(new Detection(rect, label, score, classId));
                }
            }
            else
            {
                // YOLO-like: pick the first output and try to parse [1, N, 6]
                var first = results.First();
                var arr = first.AsEnumerable<float>().ToArray();

                // Try shapes
                // Common is [1, N, 6] -> flatten to N*6
                int stride = 6;
                int n = arr.Length / stride;

                // Heuristic: sometimes YOLO gives [1, 6, N]
                if (n == 0 && arr.Length % 6 == 0)
                {
                    stride = arr.Length / 6;
                    n = arr.Length / stride;
                }

                for (int i = 0; i < n; i++)
                {
                    float x1 = arr[i * 6 + 0];
                    float y1 = arr[i * 6 + 1];
                    float x2 = arr[i * 6 + 2];
                    float y2 = arr[i * 6 + 3];
                    float score = arr[i * 6 + 4];
                    int classId = (int)arr[i * 6 + 5];
                    if (score < scoreThreshold) continue;

                    // Assume coords are model-space absolute
                    float sx = (float)bgrFrame.Cols / _inW;
                    float sy = (float)bgrFrame.Rows / _inH;
                    var rect = ClampToFrame(x1 * sx, y1 * sy, x2 * sx, y2 * sy, bgrFrame.Cols, bgrFrame.Rows);

                    string label = (classId >= 0 && classId < _labels.Count) ? _labels[classId] : $"cls{classId}";
                    dets.Add(new Detection(rect, label, score, classId));
                }
            }

            // NMS per class
            var final = NonMaxSuppression(dets, nmsThreshold);
            return final;
        }

        private static bool Has(IEnumerable<string> outputs, string namePart)
            => outputs.Any(n => n.IndexOf(namePart, StringComparison.OrdinalIgnoreCase) >= 0);

        private static System.Drawing.RectangleF ClampToFrame(float x1, float y1, float x2, float y2, int fw, int fh)
        {
            float xx1 = Math.Max(0, Math.Min(x1, x2));
            float yy1 = Math.Max(0, Math.Min(y1, y2));
            float xx2 = Math.Min(fw - 1, Math.Max(x1, x2));
            float yy2 = Math.Min(fh - 1, Math.Max(y1, y2));
            float w = Math.Max(0, xx2 - xx1);
            float h = Math.Max(0, yy2 - yy1);
            return new System.Drawing.RectangleF(xx1, yy1, w, h);
        }

        private static float IoU(System.Drawing.RectangleF a, System.Drawing.RectangleF b)
        {
            float x1 = Math.Max(a.Left, b.Left);
            float y1 = Math.Max(a.Top, b.Top);
            float x2 = Math.Min(a.Right, b.Right);
            float y2 = Math.Min(a.Bottom, b.Bottom);
            float w = Math.Max(0, x2 - x1);
            float h = Math.Max(0, y2 - y1);
            float inter = w * h;
            float union = a.Width * a.Height + b.Width * b.Height - inter + 1e-6f;
            return inter / union;
        }

        private static List<Detection> NonMaxSuppression(List<Detection> dets, float nmsThreshold)
        {
            var byClass = dets.GroupBy(d => d.ClassId);
            var outList = new List<Detection>();

            foreach (var grp in byClass)
            {
                var list = grp.OrderByDescending(d => d.Confidence).ToList();
                var keep = new List<Detection>();

                while (list.Count > 0)
                {
                    var best = list[0];
                    keep.Add(best);
                    list.RemoveAt(0);

                    list = list.Where(d => IoU(best.Rect, d.Rect) < nmsThreshold).ToList();
                }

                outList.AddRange(keep);
            }

            return outList;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }

    public record Detection(System.Drawing.RectangleF Rect, string Label, float Confidence, int ClassId);
}
