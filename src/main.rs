
use std::{io::{BufReader, BufRead}, fs::File};


fn main() {
   
    let lines: Vec<String> = BufReader::new(File::open("input.txt").unwrap()).lines().map(|l| l.unwrap()).collect();

}