# Content Aware Fill for Linux's Python

A simple baseline for image inpainting. The program will automatically fill in masked region via Content Aware Fill algorithm. An online interactive demo can be found [here](https://61315.github.io/resynthesizer/painter.html).

I borrowed from other's [repo](https://github.com/61315/resynthesizer), and I made it usable for Linux's Python.

## Build from Source

You can build it from source by:
```bash
git clone git@github.com:Karbo123/content-aware-fill.git --depth=1
cd content-aware-fill && make libcwf
```

## Usage

You can use it as follows:
```bash
cd python && ipython

# inside python, run:
import libcwf, cv2
img = cv2.imread("../assets/cat.png")
mask = cv2.imread("../assets/mask.png", 0)
out, state = libcwf.content_aware_fill(img, mask)
cv2.imwrite("../assets/cat_inpaint.png", out)
assert state == "IMAGE_SYNTH_SUCCESS"
# image have been written to: ../assets/cat_inpaint.png

```

