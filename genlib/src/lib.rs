use rand::{self, prelude::SliceRandom};
use std::cmp::Ordering;

pub trait Specimen {
  fn generate() -> Self;
  fn fitness(&self) -> f64;
  fn recombine_with(&self, rhs: &Self) -> Self;
  fn should_mutate(&self) -> bool;
  fn mutate(&mut self);
}

pub enum SpecimenOrdering {
  Any,
  Weakest,
  Strongest,
  Random,
}

pub struct Evolver<E: Specimen> {
  original_population_size: usize,
  population: Vec<E>,
  wipe_part: f64,
}

impl<E: Specimen> Evolver<E> {
  pub fn new(population_size: usize, wipe_part: f64) -> Evolver<E> {
    Evolver {
      original_population_size: population_size,
      population: Vec::new(),
      wipe_part,
    }
  }

  pub fn introduce_population(&mut self) {
    self.kill(1.0, SpecimenOrdering::Any);
    self.populate();
  }

  pub fn evolve_for(&mut self, generations_count: usize) {
    for _ in 0..generations_count {
      self.select();
      self.recombine();
      self.try_mutate();
      self.populate();
    }
  }

  pub fn evolve_until(&mut self, target_fitness: f64, generations_limit: usize) -> Option<usize> {
    for i in 0..generations_limit {
      self.select();
      self.recombine();
      self.try_mutate();
      self.populate();

      if self.has_winning_specimen(target_fitness) {
        return Some(i);
      }
    }
    None
  }

  pub fn peek_population(&mut self, part: f64, ordering: SpecimenOrdering) -> &[E] {
    self.sort(ordering);
    &self.population[0..self.count_from_percent(part)]
  }

  pub fn peek_best_specimen(&self) -> &E {
    self
      .population
      .iter()
      .max_by(|lhs, rhs| {
        if lhs.fitness() > rhs.fitness() {
          Ordering::Greater
        } else {
          Ordering::Less
        }
      })
      .unwrap()
  }

  pub fn peek_worst_specimen(&self) -> &E {
    self
      .population
      .iter()
      .max_by(|lhs, rhs| {
        if lhs.fitness() < rhs.fitness() {
          Ordering::Greater
        } else {
          Ordering::Less
        }
      })
      .unwrap()
  }

  fn select(&mut self) {
    self.kill(self.wipe_part, SpecimenOrdering::Weakest);
  }

  fn kill(&mut self, part: f64, ordering: SpecimenOrdering) {
    match ordering {
      SpecimenOrdering::Any => (),
      SpecimenOrdering::Weakest => self.sort(SpecimenOrdering::Strongest),
      SpecimenOrdering::Strongest => self.sort(SpecimenOrdering::Weakest),
      SpecimenOrdering::Random => self.sort(SpecimenOrdering::Random),
    }
    self
      .population
      .truncate(self.population.len() - self.count_from_percent(part));
  }

  fn sort(&mut self, ordering: SpecimenOrdering) {
    match ordering {
      SpecimenOrdering::Any => (),
      SpecimenOrdering::Weakest => self.population.sort_by(|lhs, rhs| {
        if lhs.fitness() > rhs.fitness() {
          Ordering::Greater
        } else {
          Ordering::Less
        }
      }),
      SpecimenOrdering::Strongest => self.population.sort_by(|lhs, rhs| {
        if lhs.fitness() < rhs.fitness() {
          Ordering::Greater
        } else {
          Ordering::Less
        }
      }),
      SpecimenOrdering::Random => self.population.shuffle(&mut rand::thread_rng()),
    }
  }

  fn recombine(&mut self) {
    self.sort(SpecimenOrdering::Random);

    let half_size = self.population.len() / 2;
    let lhs = &self.population[..half_size];
    let rhs = &self.population[half_size..];
    let mut offsprings: Vec<_> = lhs
      .iter()
      .zip(rhs.iter())
      .map(|(left, right)| left.recombine_with(right))
      .collect();

    self.population.append(&mut offsprings);
  }

  fn try_mutate(&mut self) {
    self.population.iter_mut().for_each(|specimen| {
      if specimen.should_mutate() {
        specimen.mutate();
      }
    });
  }

  fn populate(&mut self) {
    self
      .population
      .resize_with(self.original_population_size, || E::generate());
  }

  fn has_winning_specimen(&self, target_fitness: f64) -> bool {
    self.peek_best_specimen().fitness() >= target_fitness
  }

  fn count_from_percent(&self, percent: f64) -> usize {
    let count = (percent * self.population.len() as f64) as usize;
    assert!(count <= self.population.len());
    count
  }
}

#[cfg(test)]
mod tests {
  use std::cmp::min;

  use rand::{prelude::SliceRandom, Rng};

  use crate::{Evolver, Specimen};

  struct MockSpecimen {
    score: i32,
  }

  impl Specimen for MockSpecimen {
    fn generate() -> Self {
      static mut COUNTER: i32 = 0;
      unsafe {
        COUNTER += 1;
        MockSpecimen { score: COUNTER }
      }
    }

    fn fitness(&self) -> f64 {
      self.score as f64
    }

    fn recombine_with(&self, _rhs: &Self) -> Self {
      unimplemented!()
    }

    fn should_mutate(&self) -> bool {
      unimplemented!()
    }

    fn mutate(&mut self) {
      unimplemented!()
    }
  }

  #[test]
  fn best_specimen() {
    let mut evolver = Evolver::<MockSpecimen>::new(50, 0.5);
    evolver.introduce_population();
    assert_eq!(evolver.peek_best_specimen().fitness(), 50.0);
  }

  const N_QUEENS: usize = 6;

  #[derive(Debug, Default)]
  struct EightQueensSpecimen {
    queens: Vec<(usize, usize)>,
  }

  impl EightQueensSpecimen {
    pub fn print(&self) {
      let mut chessboard = vec![vec![false; N_QUEENS]; N_QUEENS];
      for queen in &self.queens {
        chessboard[queen.0][queen.1] = true;
      }

      for y in 0..N_QUEENS {
        for x in 0..N_QUEENS {
          print!("{}", if chessboard[x][y] { "X" } else { "+" });
        }
        println!();
      }
    }
  }

  impl Specimen for EightQueensSpecimen {
    fn generate() -> Self {
      let mut queens = Vec::new();
      for x in 0..N_QUEENS {
        for y in 0..N_QUEENS {
          queens.push((x, y));
        }
      }
      queens.shuffle(&mut rand::thread_rng());
      queens.truncate(N_QUEENS);

      EightQueensSpecimen { queens }
    }

    fn fitness(&self) -> f64 {
      let mut beatings = 0i32;

      for queen in &self.queens {
        beatings += self
          .queens
          .iter()
          .filter(|other| other.0 == queen.0 || other.1 == queen.1)
          .count() as i32
          - 1;

        // Diagonal magic
        let left_right_diag_x = queen.0 - min(queen.0, queen.1);
        let left_right_diag_y = queen.1 - min(queen.0, queen.1);
        let right_left_diag_x = queen.0 + min(N_QUEENS - 1 - queen.0, queen.1);
        let right_left_diag_y = queen.1 - min(N_QUEENS - 1 - queen.0, queen.1);

        let left_right_diag = (left_right_diag_x..N_QUEENS).zip(left_right_diag_y..N_QUEENS);
        let right_left_diag = (0..=right_left_diag_x)
          .rev()
          .zip(right_left_diag_y..N_QUEENS);
        for (x, y) in left_right_diag {
          beatings += self
            .queens
            .iter()
            .filter(|other| other.0 == x && other.1 == y)
            .count() as i32;
        }
        for (x, y) in right_left_diag {
          beatings += self
            .queens
            .iter()
            .filter(|other| other.0 == x && other.1 == y)
            .count() as i32;
        }
        beatings -= 2;
      }

      0.0 - (beatings as f64 / 2.0)
    }

    fn recombine_with(&self, rhs: &Self) -> Self {
      let queens = [&self.queens[..N_QUEENS / 2], &rhs.queens[N_QUEENS / 2..]].concat();
      EightQueensSpecimen { queens }
    }

    fn should_mutate(&self) -> bool {
      rand::thread_rng().gen_bool(0.5)
    }

    fn mutate(&mut self) {
      let selected_idx = rand::thread_rng().gen_range(0..N_QUEENS);

      let pos = self.queens[selected_idx];
      let new_pos = (pos.1, pos.0);

      if self
        .queens
        .iter()
        .filter(|other| other.0 == new_pos.1 && other.1 == new_pos.0)
        .count()
        == 0
      {
        self.queens[selected_idx] = new_pos;
      }
    }
  }

  #[test]
  fn eight_queens() {
    let mut evolver = Evolver::<EightQueensSpecimen>::new(100, 0.75);
    evolver.introduce_population();
    println!(
      "First generation! Best: {}, Worst: {}",
      evolver.peek_best_specimen().fitness(),
      evolver.peek_worst_specimen().fitness()
    );

    if let Some(generations) = evolver.evolve_until(-0.5, 5000) {
      println!(
        "Winning specimen found in {} generation! Best: {}, Worst: {}",
        generations,
        evolver.peek_best_specimen().fitness(),
        evolver.peek_worst_specimen().fitness()
      );
      evolver.peek_best_specimen().print();
    }

    assert_eq!(evolver.peek_best_specimen().fitness(), 0.0);
  }
}
