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

        // Detect keypoints using SIFT
        let mut sift = SIFT::create(0, 3, 0.04, 10.0, 1.6, false)?;
        let mut keypoints = Vector::<KeyPoint>::new();
        sift.detect(&frame, &mut keypoints, &core::no_array())?;

        for kp in keypoints.iter() {
            let center = kp.pt();
            let size = kp.size() as i32 / 2;
            let top_left = core::Point::new((center.x - size as f32) as i32, (center.y - size as f32) as i32);
        
            imgproc::rectangle(
                &mut frame,
                core::Rect::new(top_left.x, top_left.y, size * 2, size * 2),
                core::Scalar::new(50.0, 50.0, 50.0, 255.0),
                1,
                imgproc::LINE_8,
                0,
            )?;
        }

        // Write to output video
        writer.write(&frame)?;
    }

    println!("Output saved to: {}", output_path.display());
    Ok(())
}