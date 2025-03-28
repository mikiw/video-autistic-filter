use opencv::{
    core, features2d::{self, SIFT}, imgproc, prelude::*, types, videoio, Result
};
use std::{env, path::Path};
use opencv::core::Vector;
use opencv::core::KeyPoint;

fn main() -> Result<()> {
    // Get input path from CLI argument
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <input_video_path>", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let suffix = "-autistic";

    // Open input video
    let mut cap = videoio::VideoCapture::from_file(input_path, videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        panic!("Cannot open video file: {}", input_path);
    }

    // Get original video properties
    let fps = cap.get(videoio::CAP_PROP_FPS)?;
    let frame_width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let frame_height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    let codec = cap.get(videoio::CAP_PROP_FOURCC)? as i32;

    println!("FPS: {}, Size: {}x{}", fps, frame_width, frame_height);

    // Build output file path
    let path = Path::new(input_path);
    let stem = path.file_stem().unwrap().to_string_lossy();
    let ext = path.extension().unwrap_or_default().to_string_lossy();
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let output_filename = format!("{}{}.{}", stem, suffix, ext);
    let output_path = parent.join(output_filename);

    // Open output video writer
    let mut writer = videoio::VideoWriter::new(
        output_path.to_str().unwrap(),
        codec,
        fps,
        core::Size::new(frame_width, frame_height),
        true,
    )?;
    if !writer.is_opened()? {
        panic!("Cannot open output video writer");
    }

    // Frame-by-frame processing
    loop {
        let mut frame = Mat::default();
        cap.read(&mut frame)?;
        if frame.empty() {
            break;
        }

        // Detect keypoints using SIFT [Recommended setting is 0.04]
        let mut sift = SIFT::create(0, 3, 0.04, 10.0, 1.6, false)?;
        let mut keypoints = Vector::<KeyPoint>::new();
        sift.detect(&frame, &mut keypoints, &core::no_array())?;
        
        // Filter keypoints by minimum size
        let min_size = 30.0; // Adjust this threshold as needed [Recommended setting]
        let filtered_keypoints: Vector<KeyPoint> = keypoints.iter()
            .filter(|kp| kp.size() >= min_size)
            .collect();

        for (i, kp) in filtered_keypoints.iter().enumerate() {
            let center = kp.pt();
            let size = kp.size() as i32 / 2;
            let top_left = core::Point::new((center.x - size as f32) as i32, (center.y - size as f32) as i32);
        
            // Define rectangle
            let rect = core::Rect::new(
                top_left.x.max(0),
                top_left.y.max(0),
                (size * 2).min(frame_width - top_left.x),
                (size * 2).min(frame_height - top_left.y),
            );

            // Increase contrast and invert colors
            if rect.width > 0 && rect.height > 0 {
                let mut roi = Mat::roi_mut(&mut frame, rect)?;
                
                // Adjust contrast
                let contrast_percent = 50.0; // Change this value from 0.0 to 100.0
                let alpha = 1.0 + (contrast_percent / 100.0); // Contrast factor
                let beta = 0.0; // Brightness offset
                core::convert_scale_abs(&roi.try_clone()?, &mut roi, alpha, beta)?;

                // Invert colors after contrast boost
                core::bitwise_not(&roi.try_clone()?, &mut roi, &core::no_array())?;
            }

            // Draw rectangle border
            imgproc::rectangle(
                &mut frame,
                rect,
                core::Scalar::new(50.0, 50.0, 50.0, 255.0),
                2,
                imgproc::LINE_8,
                0,
            )?;

            // Draw ID in top-left corner of rectangle
            imgproc::put_text(
                &mut frame,
                &i.to_string(), // convert index to string
                top_left, // position
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.9, // font scale
                core::Scalar::new(255.0, 255.0, 255.0, 255.0), // white color
                2, // thickness
                imgproc::LINE_AA,
                false,
            )?;
        }

        // Connect all keypoints that are within a certain distance threshold
        let distance_threshold = 300.0; // [Recommended setting]
        for (i, kp1) in filtered_keypoints.iter().enumerate() {
            let pt1 = kp1.pt();
            for (j, kp2) in filtered_keypoints.iter().enumerate() {
                if i >= j {
                    continue;
                }
                let pt2 = kp2.pt();
                let dx = pt1.x - pt2.x;
                let dy = pt1.y - pt2.y;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist <= distance_threshold {
                    imgproc::line(
                        &mut frame,
                        core::Point::new(pt1.x as i32, pt1.y as i32),
                        core::Point::new(pt2.x as i32, pt2.y as i32),
                        core::Scalar::new(200.0, 200.0, 200.0, 255.0),
                        1,
                        imgproc::LINE_AA,
                        0,
                    )?;
                }
            }
        }

        // Write to output video
        writer.write(&frame)?;
    }

    println!("Output saved to: {}", output_path.display());
    Ok(())
}