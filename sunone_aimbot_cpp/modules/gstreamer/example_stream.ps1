$Host_ip = "192.168.66.20"
$Port = 5000
$CropSize = 640
$Fps = 60
$BitrateKbps = 2000
$ShowCursor
$ScreenW = 2560
$ScreenH = 1440

$x = [int](($ScreenW - $CropSize) / 2)
$y = [int](($ScreenH - $CropSize) / 2)

$gst = (Get-Command "gst-launch-1.0.exe" -ErrorAction SilentlyContinue).Source
if (-not $gst) {
    $fallback = "C:\Program Files\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0.exe"
    if (Test-Path $fallback) { $gst = $fallback }
    else { throw "gst-launch-1.0.exe not found. Add GStreamer bin to PATH or edit fallback path in script." }
}

$cursorVal = if ($ShowCursor) { "true" } else { "false" }

Write-Host "Streaming center crop ${CropSize}x${CropSize} from ${ScreenW}x${ScreenH} (x=$x, y=$y) -> udp://$Host_ip`:$Port"

$args = @(
    "-v",
    "d3d11screencapturesrc", "show-cursor=$cursorVal", "crop-x=$x", "crop-y=$y", "crop-width=$CropSize", "crop-height=$CropSize",
    "!", "queue", "max-size-buffers=1", "leaky=downstream",
    "!", "d3d11download",
    "!", "videoconvert",
    "!", "video/x-raw,format=I420,framerate=$Fps/1",
    "!", "x264enc", "tune=zerolatency", "speed-preset=ultrafast", "bitrate=$BitrateKbps", "key-int-max=30", "bframes=0",
    "!", "rtph264pay", "config-interval=1", "pt=96", "mtu=1200",
    "!", "udpsink", "host=$Host_ip", "port=$Port", "sync=false"
)

& $gst @args
