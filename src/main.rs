
use std::{io::{BufReader, BufRead}, fs::File, collections::{HashMap}, hash::{Hash, Hasher}, cmp::{min, max}};
use ndarray::{Array2, Axis, Array1};


struct OrderAmbivalentPair<T: Ord>(T, T);
impl<T: Ord + Hash> Hash for OrderAmbivalentPair<T> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        min(&self.0, &self.1).hash(hasher);
        max(&self.0, &self.1).hash(hasher);
    }
}

impl<T: PartialEq + Ord> PartialEq for OrderAmbivalentPair<T> {
    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }

    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1 || self.0 == other.1 && self.1 == other.0
    }
}

impl<T: PartialEq + Ord> Eq for OrderAmbivalentPair<T> {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Index {
    row: usize,
    col: usize
}

impl From<(usize, usize)> for Index {
    fn from(value: (usize, usize)) -> Self {
        Self { row: value.0, col: value.1 }
    }
}

impl Into<(usize, usize)> for Index {
    fn into(self) -> (usize, usize) {
        (self.row, self.col)
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Galaxy {
    id: usize,
    index: Index
}

impl Ord for Galaxy {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for Galaxy {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for Galaxy {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

struct Incrementer {
    state: usize
}

impl Incrementer {

    fn new() -> Self {
        Self{state: 0}
    }

    fn get_id(&mut self) -> usize {
        let id = self.state;
        self.state += 1;
        id
    }
}

#[derive(Debug, Clone)]
struct Universe {
    map: Array2<Option<Galaxy>>
}

impl Universe {

    fn expand_from_map(&mut self) {
    
        let mut stage_one = Array2::default((0, self.map.shape()[1]));
        let new_row = Array1::default(self.map.shape()[1]);
    
        for row in self.map.axis_iter(Axis(0)) {
            if row.iter().any(|e| e.is_some()) {
                stage_one.push_row(row).unwrap();
            } else {
                stage_one.push_row(new_row.view()).unwrap();
                stage_one.push_row(new_row.view()).unwrap();
            }
        }
    
        let mut stage_two = Array2::default((stage_one.shape()[0], 0));
        let new_col = Array1::default(stage_one.shape()[0]);
    
        for col in stage_one.axis_iter(Axis(1)) {
            if col.iter().any(|e| e.is_some()) {
                stage_two.push_column(col).unwrap();
            } else {
                stage_two.push_column(new_col.view()).unwrap();
                stage_two.push_column(new_col.view()).unwrap();
            }
        }

        stage_two.indexed_iter_mut().for_each(|((row, col), element)| {
            if let Some(galaxy) = element {
                galaxy.index = Index{row, col}
            }
        });
    
        self.map = stage_two
    }
}

impl From<Vec<String>> for Universe {

    fn from(lines: Vec<String>) -> Self {

        let mut incrementer = Incrementer::new();

        let shape = (lines.len(), lines[0].len());
        let elements: Array2<Option<Galaxy>> = lines.iter().enumerate().map(|(row, line)| {
            let row: Array1<Option<Galaxy>> = Array1::from_iter(line.chars().enumerate().map(|(col, c)| {
                match c {
                    '.' => Option::None,
                    '#' => Option::Some(Galaxy{id: incrementer.get_id(), index: Index{row, col}}),
                    _ => panic!("Error parsing input"),
                }
            }));
            return row;
        }).fold(Array2::default((0, shape.1)), |mut table, row| {
            table.push_row(row.view()).unwrap();
            return table;
        });
        
        Self { map: elements }
    }
}

fn locate_galaxies(universe: &Universe) -> Vec<Galaxy> {
    universe.map.iter().filter_map(|e| *e).collect()
}

fn expand_from_galaxies(galaxies: &Vec<Galaxy>, expansion_factor: usize) -> Vec<Galaxy> {
    let mut res = galaxies.clone();

    let min_row = galaxies.iter().min_by(|a, b| a.index.row.cmp(&b.index.row)).unwrap().index.row;
    let max_row = galaxies.iter().max_by(|a, b| a.index.row.cmp(&b.index.row)).unwrap().index.row;
    let min_col = galaxies.iter().min_by(|a, b| a.index.col.cmp(&b.index.col)).unwrap().index.col;
    let max_col = galaxies.iter().max_by(|a, b| a.index.col.cmp(&b.index.col)).unwrap().index.col;

    let mut running_row = min_row;

    for _ in min_row..=max_row {

        if res.iter().any(|galaxy| galaxy.index.row == running_row) {
            running_row += 1;
        } else {
            res.iter_mut().filter(|galaxy| galaxy.index.row > running_row).for_each(|galaxy| galaxy.index.row += expansion_factor - 1);
            running_row += expansion_factor;
        }
    }

    let mut running_col = min_col;

    for _ in min_col..=max_col {

        if res.iter().any(|galaxy| galaxy.index.col == running_col) {
            running_col += 1;
        } else {
            res.iter_mut().filter(|galaxy| galaxy.index.col > running_col).for_each(|galaxy| galaxy.index.col += expansion_factor - 1);
            running_col += expansion_factor;
        }
    }

    res
}

fn extract_shortest_paths(galaxies: &Vec<Galaxy>) -> HashMap::<OrderAmbivalentPair<Galaxy>, usize> {

    let mut res = HashMap::<OrderAmbivalentPair<Galaxy>, usize>::new();

    for (index1, galaxy1) in galaxies[0..galaxies.len() - 1].iter().enumerate() {
        for galaxy2 in galaxies[index1..].iter() {
            let shortest_path = calculate_shortest_path(galaxy1, galaxy2);
            res.insert(OrderAmbivalentPair(*galaxy1, *galaxy2), shortest_path);
        }
    }
    res
}

fn calculate_shortest_path(galaxy1: &Galaxy, galaxy2: &Galaxy) -> usize {
    galaxy1.index.row.abs_diff(galaxy2.index.row) + galaxy1.index.col.abs_diff(galaxy2.index.col)
}

fn part1(mut universe: Universe) -> usize {
    universe.expand_from_map();

    let galaxies = locate_galaxies(&universe);
    let shortest_paths = extract_shortest_paths(&galaxies);

    shortest_paths.values().sum()
}

fn part2(universe: Universe) -> usize {

    let galaxies = locate_galaxies(&universe);
    let expanded_galaxies = expand_from_galaxies(&galaxies, 1000000);
    let shortest_paths = extract_shortest_paths(&expanded_galaxies);

    shortest_paths.values().sum()
}


fn main() {
   
    let lines: Vec<String> = BufReader::new(File::open("input.txt").unwrap()).lines().map(|l| l.unwrap()).collect();
    let universe: Universe = lines.into();

    println!("Part 1 {}", part1(universe.clone()));
    println!("Part 2 {}", part2(universe));
}