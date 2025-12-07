use super::{Error, ReedSolomon};
use rand::{distr, random};
use std::{iter, sync::LazyLock};

/// Fixture: reusable test data for all reconstruction variants
struct TestData {
    reed_solomon: ReedSolomon,
    shards: Vec<Vec<u8>>,
}

static FIXTURE: LazyLock<TestData> = LazyLock::new(|| {
    const SHARD_LEN: usize = 20;

    let reed_solomon = ReedSolomon::new(8, 5).unwrap();
    let mut shards = make_random_shards(SHARD_LEN, 13);
    reed_solomon.encode(&mut shards).unwrap();

    TestData {
        reed_solomon,
        shards,
    }
});

fn make_random_shards<T>(shard_len: usize, num_shards: usize) -> Vec<Vec<T>>
where
    distr::StandardUniform: distr::Distribution<T>,
{
    let make_shard = || iter::repeat_with(random::<T>).take(shard_len).collect();
    iter::repeat_with(make_shard).take(num_shards).collect()
}

fn shards_to_option_shards<T: Clone>(shards: &[Vec<T>]) -> Vec<Option<Vec<T>>> {
    shards.iter().cloned().map(Some).collect()
}

fn option_shards_to_shards<T: Clone>(shards: &[Option<Vec<T>>]) -> Vec<Vec<T>> {
    shards
        .iter()
        .cloned()
        .map(|opt| opt.expect("Missing shard"))
        .collect()
}

/*
#[test]
fn test_reconstruct_shards1() {
    let t = &*FIXTURE;
    let mut option_shards = shards_to_option_shards(&t.shards);

    // Try to decode with all shards present
    t.reed_solomon.reconstruct(&mut option_shards).unwrap();
    let shards = option_shards_to_shards(&option_shards);
    assert_eq!(&shards, &t.shards);
    assert!(t.reed_solomon.verify(&shards).unwrap());
    assert_eq!(&shards, &t.shards);
}

#[test]
fn test_reconstruct_shards2() {
    let t = &*FIXTURE;
    let mut option_shards = shards_to_option_shards(&t.shards);

    option_shards[0] = None;
    option_shards[2] = None;
    t.reed_solomon.reconstruct(&mut option_shards).unwrap();
    let shards = option_shards_to_shards(&option_shards);
    assert!(t.reed_solomon.verify(&shards).unwrap());
    assert_eq!(&shards, &t.shards);
}

#[test]
fn test_reconstruct_shards3() {
    let t = &*FIXTURE;
    let mut option_shards = shards_to_option_shards(&t.shards);

    option_shards[0] = None;
    option_shards[2] = None;
    option_shards[7] = None;
    t.reed_solomon.reconstruct(&mut option_shards).unwrap();
    let shards = option_shards_to_shards(&option_shards);
    assert!(t.reed_solomon.verify(&shards).unwrap());
    assert_eq!(&shards, &t.shards);
}
*/

#[test]
fn test_reconstruct_shards4() {
    let t = &*FIXTURE;
    let mut option_shards = shards_to_option_shards(&t.shards);

    // Try to decode with 6 data and 4 parity shards
    option_shards[0] = None;
    option_shards[2] = None;
    option_shards[12] = None;
    t.reed_solomon.reconstruct(&mut option_shards).unwrap();
    let shards = option_shards_to_shards(&option_shards);
    assert!(t.reed_solomon.verify(&shards).unwrap());
    assert_eq!(&shards, &t.shards);
}

#[test]
fn test_reconstruct_shards5() {
    let t = &*FIXTURE;
    let mut option_shards = shards_to_option_shards(&t.shards);

    // Try to reconstruct data only
    option_shards[0] = None;
    option_shards[1] = None;
    option_shards[12] = None;
    // FIXME: r.reconstruct_data(&mut shards).unwrap();
    t.reed_solomon.reconstruct(&mut option_shards).unwrap();
    let data_shards = option_shards_to_shards(&option_shards[0..8]);
    assert_eq!(t.shards[0], data_shards[0]);
    assert_eq!(t.shards[1], data_shards[1]);
    assert_eq!(None, option_shards[12]);
}

/*
#[test]
fn test_reconstruct_shards6() {
    let t = &*FIXTURE;
    let mut option_shards = shards_to_option_shards(&t.shards);

    // Try to decode with 7 data and 1 parity shards
    option_shards[0] = None;
    option_shards[1] = None;
    option_shards[9] = None;
    option_shards[10] = None;
    option_shards[11] = None;
    option_shards[12] = None;
    assert_eq!(
        t.reed_solomon.reconstruct(&mut option_shards).unwrap_err(),
        Error::TooFewShardsPresent
    );
}
*/
