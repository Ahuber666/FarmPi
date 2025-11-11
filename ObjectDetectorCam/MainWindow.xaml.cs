using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.WpfExtensions;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace ObjectDetectorCam
{
    public partial class MainWindow : System.Windows.Window
    {
        private VideoCapture? _capture;
        private CancellationTokenSource? _cts;
        private InferenceEngine? _engine;

        // ---- Customize these if needed ----
        private readonly string _modelPath = System.IO.Path.Combine("model", "model.onnx");
        private readonly string _labelsPath = System.IO.Path.Combine("model", "labels.txt");
        private readonly int _cameraIndex = 0;
        private readonly float _scoreThreshold = 0.40f;
        private readonly float _nmsThreshold = 0.45f;
        // If you know your model input size, set it here (e.g., 320, 320 or 640, 640).
        // The engine will discover it automatically, but setting explicitly can avoid one warmup pass.
        private readonly (int W, int H)? _knownInputWH = null; // e.g. (640, 640)
        // -----------------------------------

        public MainWindow()
        {
            InitializeComponent();
            this.Loaded += MainWindow_Loaded;
            this.Closed += MainWindow_Closed;
            this.KeyDown += (s, e) => { if (e.Key == System.Windows.Input.Key.Escape) Close(); };
        }

        private async void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            try
            {
                _engine = new InferenceEngine(_modelPath, _labelsPath, _knownInputWH);
            }
            catch (System.Exception ex)
            {
                MessageBox.Show($"Failed to initialize ONNX model:\n{ex.Message}", "Error",
                    MessageBoxButton.OK, MessageBoxImage.Error);
                Close();
                return;
            }

            _capture = new VideoCapture(_cameraIndex);
            if (!_capture.IsOpened())
            {
                MessageBox.Show("Could not open webcam.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                Close();
                return;
            }

            _cts = new CancellationTokenSource();
            StatusText.Text = $"Model: {System.IO.Path.GetFileName(_modelPath)} — Camera: {_cameraIndex}";

            await Task.Run(() => CaptureLoop(_cts.Token));
        }

        private void MainWindow_Closed(object? sender, System.EventArgs e)
        {
            try
            {
                _cts?.Cancel();
                _capture?.Release();
                _capture?.Dispose();
                _engine?.Dispose();
            }
            catch { /* ignore */ }
        }

        private void CaptureLoop(CancellationToken token)
        {
            using var frame = new Mat();

            // Try to match window size initially
            this.Dispatcher.Invoke(() =>
            {
                Overlay.Width = this.ActualWidth;
                Overlay.Height = this.ActualHeight;
            });

            while (!token.IsCancellationRequested)
            {
                if (!(_capture?.Read(frame) ?? false) || frame.Empty())
                    continue;

                // Run inference (downscaled internally)
                var detections = _engine!.Detect(frame, _scoreThreshold, _nmsThreshold);

                // Convert to WPF image source
                BitmapSource bitmap = frame.ToBitmapSource();
                bitmap.Freeze();

                // Render on UI thread
                this.Dispatcher.Invoke(() =>
                {
                    VideoImage.Source = bitmap;

                    // Make overlay the same pixel size as the displayed image
                    // We map boxes in "frame pixel" space to the control by using the Image's actual displayed size.
                    var img = VideoImage;
                    var bmpW = bitmap.PixelWidth;
                    var bmpH = bitmap.PixelHeight;

                    var displayW = img.ActualWidth;
                    var displayH = img.ActualHeight;

                    // The Image uses Uniform stretch: compute letterboxing offsets and scale.
                    double scale = System.Math.Min(displayW / bmpW, displayH / bmpH);
                    double renderW = bmpW * scale;
                    double renderH = bmpH * scale;
                    double offsetX = (displayW - renderW) / 2.0;
                    double offsetY = (displayH - renderH) / 2.0;

                    Overlay.Width = displayW;
                    Overlay.Height = displayH;
                    Overlay.Children.Clear();

                    foreach (var det in detections)
                    {
                        // Map from frame pixels to display pixels
                        double x = offsetX + det.Rect.X * scale;
                        double y = offsetY + det.Rect.Y * scale;
                        double w = det.Rect.Width * scale;
                        double h = det.Rect.Height * scale;

                        var r = new Rectangle
                        {
                            Width = w,
                            Height = h,
                            Stroke = Brushes.Lime,
                            StrokeThickness = 2,
                            Fill = new SolidColorBrush(Color.FromArgb(30, 0, 255, 0))
                        };
                        Canvas.SetLeft(r, x);
                        Canvas.SetTop(r, y);
                        Overlay.Children.Add(r);

                        var labelText = $"{det.Label} {(det.Confidence * 100):0}%";
                        var tb = new System.Windows.Controls.Border
                        {
                            Background = Brushes.Black,
                            CornerRadius = new CornerRadius(4),
                            Opacity = 0.75,
                            Child = new System.Windows.Controls.TextBlock
                            {
                                Text = labelText,
                                Foreground = Brushes.White,
                                FontWeight = FontWeights.SemiBold,
                                Margin = new Thickness(6, 2, 6, 2)
                            }
                        };
                        Canvas.SetLeft(tb, x);
                        Canvas.SetTop(tb, y - 24 < 0 ? 0 : y - 24);
                        Overlay.Children.Add(tb);
                    }

                    StatusText.Text = $"Detections: {detections.Count}";
                });
            }
        }
    }
}
