//! Pixel-art upscaling for the interface art.
//!
//! Scale2x (EPX) doubles an image by looking at each pixel's four
//! neighbours: diagonal edges come out smooth while flat areas, outlines and
//! text stay crisp, which bilinear filtering alone would blur. It runs on
//! palette indices, so transparency (index 0) and colours are exact.

use crate::image::Image;

/// One Scale2x pass.
pub fn scale2x(image: &Image) -> Image {
    let (w, h) = (image.width, image.height);
    let at = |x: isize, y: isize| -> u8 {
        let x = x.clamp(0, w as isize - 1) as usize;
        let y = y.clamp(0, h as isize - 1) as usize;
        image.pixels[y * w + x]
    };
    let mut out = vec![0u8; w * h * 4];
    for y in 0..h as isize {
        for x in 0..w as isize {
            let p = at(x, y);
            let (a, b, c, d) = (at(x, y - 1), at(x + 1, y), at(x - 1, y), at(x, y + 1));
            let mut e = [p; 4];
            if c == a && c != d && a != b {
                e[0] = a;
            }
            if a == b && a != c && b != d {
                e[1] = b;
            }
            if d == c && d != b && c != a {
                e[2] = c;
            }
            if b == d && b != a && d != c {
                e[3] = d;
            }
            let (ox, oy) = (x as usize * 2, y as usize * 2);
            out[oy * w * 2 + ox] = e[0];
            out[oy * w * 2 + ox + 1] = e[1];
            out[(oy + 1) * w * 2 + ox] = e[2];
            out[(oy + 1) * w * 2 + ox + 1] = e[3];
        }
    }
    Image {
        width: w * 2,
        height: h * 2,
        flags: image.flags,
        anchor: image.anchor.map(|v| v * 2),
        pixels: out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonals_smooth_and_flat_areas_stay() {
        // A 2x2 diagonal: 1 0 / 0 1.
        let image = Image {
            width: 2,
            height: 2,
            flags: 0,
            anchor: [0; 4],
            pixels: vec![1, 0, 0, 1],
        };
        let big = scale2x(&image);
        assert_eq!((big.width, big.height), (4, 4));
        // The top-left pixel stays 1 at its own corner.
        assert_eq!(big.pixels[0], 1);
        let flat = Image {
            pixels: vec![5; 4],
            ..image
        };
        assert!(scale2x(&flat).pixels.iter().all(|&p| p == 5));
    }
}
