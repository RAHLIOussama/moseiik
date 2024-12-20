#[cfg(test)]
mod tests {
    use std::process::Command;
    use std::path::Path;
    use image::{GenericImageView, DynamicImage};
    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_x86() {
        let output = Command::new("cargo")
            .args(&["run", "--release", "--", "--image", "assets/kit.jpeg", "--tiles", "assets/images", "--output", "out.png", "--tile-size", "25"])
            .output()
            .expect("Failed to execute command");

        assert!(output.status.success(), "Failed to execute compute_mosaic: {:?}", output);

        let generated_image_path = "out.png";
        let ground_truth_image_path = "assets/ground-truth-kit.png";

        let generated_image = image::open(generated_image_path).expect("Failed to open generated image");
        let ground_truth_image = image::open(ground_truth_image_path).expect("Failed to open ground truth image");

        // test avx2 or sse2 if available
        assert_eq!(
            generated_image.dimensions(),
            ground_truth_image.dimensions(),
            "Dimensions of the generated image do not match the ground truth"
        );

        /*if Path::new(generated_image_path).exists() {
            std::fs::remove_file(generated_image_path).unwrap();
        }*/
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_aarch64() {
    let output = Command::new("cargo")
            .args(&["run", "--release", "--", "--image", "assets/kit.jpeg", "--tiles", "assets/images", "--output", "out.png", "--tile-size", "25"])
            .output()
            .expect("Failed to execute command");

        assert!(output.status.success(), "Failed to execute compute_mosaic: {:?}", output);

        let generated_image_path = "out.png";
        let ground_truth_image_path = "assets/ground-truth-kit.png";

        let generated_image = image::open(generated_image_path).expect("Failed to open generated image");
        let ground_truth_image = image::open(ground_truth_image_path).expect("Failed to open ground truth image");

        assert_eq!(
            generated_image.dimensions(),
            ground_truth_image.dimensions(),
            "Dimensions of the generated image do not match the ground truth"
        );

        /*if Path::new(generated_image_path).exists() {
            std::fs::remove_file(generated_image_path).unwrap();
        }*/
    }

    #[test]
    fn test_generic() {
        let output = Command::new("cargo")
            .args(&["run", "--release", "--", "--image", "assets/kit.jpeg", "--tiles", "assets/images", "--output", "out.png", "--tile-size", "25"])
            .output()
            .expect("Failed to execute command");

        assert!(output.status.success(), "Failed to execute compute_mosaic: {:?}", output);

        let generated_image_path = "out.png";
        let ground_truth_image_path = "assets/ground-truth-kit.png";

        let generated_image = image::open(generated_image_path).expect("Failed to open generated image");
        let ground_truth_image = image::open(ground_truth_image_path).expect("Failed to open ground truth image");

        assert!(
            generated_image == ground_truth_image,
            "The generated image does not match the ground truth"
        );

        /*if Path::new(generated_image_path).exists() {
            std::fs::remove_file(generated_image_path).unwrap();
        }*/
    }
}
