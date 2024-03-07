use std::time::Instant;

use clap::{arg, value_parser, Command};
use snescompress::{compress, decompress, HAL, LZ2, LZ3};

enum Algo {
    LZ2,
    LZ3,
    HAL,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("thingy")
        .arg(
            arg!(-a --algorithm <ALG> "Compression algorithm to use")
                .default_value("lz3")
                .value_parser(["lz2", "lz3", "hal"]),
        )
        .subcommand(
            Command::new("decompress")
                .alias("decomp")
                .arg(arg!(<in> "input filename"))
                .arg(arg!(<out> "output (compressed) filename"))
                .arg(arg!(-b --batch "treat inputs as directories, decompress every file in <in> to a file in <out>")),
        )
        .subcommand(
            Command::new("compress")
                .alias("comp")
                .arg(arg!(<in> "input filename"))
                .arg(arg!(<out> "output (decompressed) filename"))
                .arg(arg!(-b --batch "treat inputs as directories, compress every file in <in> to a file in <out>")),
        )
        .subcommand(
            Command::new("bench-compress")
                .arg(arg!(<dir>))
                .arg(
                    arg!(-i --iters <iters>)
                        .default_value("1")
                        .value_parser(value_parser!(usize)),
                )
                .arg(arg!(-d --check_dir <dir>)),
        )
        .subcommand_required(true)
        .get_matches();

    let algo: &str = (matches.get_one::<String>("algorithm"))
        .map(|x| &**x)
        .unwrap_or("lz3");

    let algo = match algo {
        "lz2" => Algo::LZ2,
        "lz3" => Algo::LZ3,
        "hal" => Algo::HAL,
        _ => unreachable!(),
    };
    // these we can't just pass the LZ number into compress directly, because we
    // need to make sure the variants are monomorphized properly. but i also
    // didn't want to do the string-parsing everywhere all the time
    let compress_one = |inp: &[u8]| -> Vec<u8> {
        let mut out = vec![];
        match algo {
            Algo::LZ2 => compress::<LZ2>(inp, 0, &mut out),
            Algo::LZ3 => compress::<LZ3>(inp, 0, &mut out),
            Algo::HAL => compress::<HAL>(inp, 0, &mut out),
        }
        out
    };
    let decompress_one = |inp: &[u8]| -> Vec<u8> {
        let mut out = vec![];
        match algo {
            Algo::LZ2 => decompress::<LZ2>(inp, &mut out),
            Algo::LZ3 => decompress::<LZ3>(inp, &mut out),
            Algo::HAL => decompress::<HAL>(inp, &mut out),
        }
        out
    };

    match matches.subcommand() {
        Some(("decompress", matches)) => {
            let inp = matches.get_one::<String>("in").unwrap();
            let out = matches.get_one::<String>("out").unwrap();
            if matches.get_flag("batch") {
                for entry in std::fs::read_dir(inp)? {
                    let entry = entry?;
                    let indata = std::fs::read(entry.path())?;
                    let outdata = decompress_one(&indata);
                    let thingy = std::path::Path::new(out).join(entry.file_name());
                    std::fs::write(thingy, outdata)?;
                }
            } else {
                let inp_file = std::fs::read(inp)?;
                let outbuf = decompress_one(&inp_file);
                std::fs::write(out, outbuf)?;
            }
        }
        Some(("compress", matches)) => {
            let inp = matches.get_one::<String>("in").unwrap();
            let out = matches.get_one::<String>("out").unwrap();
            if matches.get_flag("batch") {
                for entry in std::fs::read_dir(inp)? {
                    let entry = entry?;
                    let indata = std::fs::read(entry.path())?;
                    let outdata = compress_one(&indata);
                    let thingy = std::path::Path::new(out).join(entry.file_name());
                    std::fs::write(thingy, outdata)?;
                }
            } else {
                let inp_file = std::fs::read(inp)?;
                let outbuf = compress_one(&inp_file);
                std::fs::write(out, outbuf)?;
            }
        }
        Some(("bench-compress", matches)) => {
            let iters = *matches.get_one::<usize>("iters").unwrap();
            let dir = matches.get_one::<String>("dir").unwrap();
            let checkd = matches.get_one::<String>("check_dir");
            let mut files = vec![];
            for entry in std::fs::read_dir(dir)? {
                files.push(entry?.path());
            }
            files.sort();
            for iter in 0..iters {
                if iter > 0 {
                    println!("--- iteration {} ---", iter + 1);
                }
                for f in &files {
                    let inp = std::fs::read(f)?;
                    let now = Instant::now();
                    let out = compress_one(&inp);
                    let duration = now.elapsed();
                    let fname = f.file_name().unwrap().to_string_lossy();
                    println!("File {} len = {}, took {:.2?}", fname, out.len(), duration);
                    let decomp = decompress_one(&out);
                    assert!(decomp == inp, "Decompression output mismatch!!");
                    if let Some(d) = checkd {
                        let thingy = std::path::Path::new(d).join(f.file_name().unwrap());
                        let expected_size = std::fs::metadata(thingy)?.len();
                        assert!(out.len() == expected_size as usize,
                            "unexpected compressed file size! expected {}", expected_size);
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    Ok(())
}
