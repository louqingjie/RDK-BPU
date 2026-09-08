// 大恒相机最小采集程序：抓 N 帧 RGB24 存为 raw 文件
// 编译: g++ -O2 -I<gx_include> capture.cpp -lgxiapi -o capture
// 运行: ./capture <帧数> <输出目录>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <vector>

#include "GxIAPI.h"
#include "DxImageProc.h"

static double now_ms() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, char **argv) {
  int n_frames = (argc > 1) ? atoi(argv[1]) : 10;
  const char *outdir = (argc > 2) ? argv[2] : "/tmp/cap";
  mkdir(outdir, 0755);

  GX_STATUS s = GXInitLib();
  if (s != GX_STATUS_SUCCESS) { printf("GXInitLib failed: 0x%x\n", s); return 1; }

  uint32_t dev_num = 0;
  GXUpdateDeviceList(&dev_num, 1000);
  if (dev_num == 0) { printf("NO_DEVICE\n"); GXCloseLib(); return 2; }
  printf("found %u device(s)\n", dev_num);

  GX_DEV_HANDLE h = nullptr;
  s = GXOpenDeviceByIndex(1, &h);
  if (s != GX_STATUS_SUCCESS) { printf("open failed: 0x%x\n", s); GXCloseLib(); return 3; }

  char model[128] = {0}, sn[128] = {0};
  size_t sz = sizeof(model);
  GXGetString(h, GX_STRING_DEVICE_MODEL_NAME, model, &sz);
  sz = sizeof(sn);
  GXGetString(h, GX_STRING_DEVICE_SERIAL_NUMBER, sn, &sz);
  printf("model=%s sn=%s\n", model, sn);

  int64_t w = 0, hgt = 0, payload = 0, bayer = 0;
  GXGetInt(h, GX_INT_WIDTH, &w);
  GXGetInt(h, GX_INT_HEIGHT, &hgt);
  GXGetInt(h, GX_INT_PAYLOAD_SIZE, &payload);
  bool has_bayer = false;
  if (GXIsImplemented(h, GX_ENUM_PIXEL_COLOR_FILTER, &has_bayer) == GX_STATUS_SUCCESS && has_bayer)
    GXGetEnum(h, GX_ENUM_PIXEL_COLOR_FILTER, &bayer);
  printf("resolution=%ldx%ld payload=%ld bayer=%ld\n", w, hgt, payload, bayer);

  GXSetEnum(h, GX_ENUM_ACQUISITION_MODE, GX_ACQ_MODE_CONTINUOUS);

  std::vector<uint8_t> raw(payload), rgb((size_t)w * hgt * 3);

  s = GXStreamOn(h);
  if (s != GX_STATUS_SUCCESS) { printf("stream on failed: 0x%x\n", s); GXCloseDevice(h); GXCloseLib(); return 4; }

  char meta[256];
  snprintf(meta, sizeof(meta), "%s/meta.txt", outdir);
  FILE *fm = fopen(meta, "w");
  fprintf(fm, "%ld %ld %d\n", w, hgt, n_frames);

  int saved = 0;
  double t0 = now_ms();
  while (saved < n_frames) {
    GX_FRAME_DATA frame;
    memset(&frame, 0, sizeof(frame));
    frame.pImgBuf = raw.data();
    frame.nImgSize = payload;
    if (GXGetImage(h, &frame, 500) != GX_STATUS_SUCCESS) { printf("get image timeout\n"); break; }
    if (frame.nStatus != GX_FRAME_STATUS_SUCCESS) continue;

    DxRaw8toRGB24Ex(frame.pImgBuf, rgb.data(),
                    (uint32_t)frame.nWidth, (uint32_t)frame.nHeight,
                    RAW2RGB_NEIGHBOUR3, (DX_PIXEL_COLOR_FILTER)bayer, false,
                    DX_ORDER_RGB);
    char path[512];
    snprintf(path, sizeof(path), "%s/frame_%03d.raw", outdir, saved);
    FILE *f = fopen(path, "wb");
    fwrite(rgb.data(), 1, (size_t)w * hgt * 3, f);
    fclose(f);
    saved++;
    if (saved == 1) printf("first frame %.1f ms after stream on\n", now_ms() - t0);
  }
  printf("captured %d frames in %.1f ms (%.1f fps)\n",
         saved, now_ms() - t0, saved * 1000.0 / (now_ms() - t0 + 1e-9));
  fprintf(fm, "captured %d\n", saved);
  fclose(fm);

  GXStreamOff(h);
  GXCloseDevice(h);
  GXCloseLib();
  return 0;
}
