use super::{Error, ReedSolomon};
use rand::{distr, random};
use std::iter;

fn make_random_shards<T: Default>(shard_len: usize, num_shards: usize) -> Vec<Vec<T>>
where
    distr::StandardUniform: distr::Distribution<T>,
{
    let make_shard = || iter::repeat_with(random::<T>).take(shard_len).collect();
    iter::repeat_with(make_shard).take(num_shards).collect()
}

fn shards_to_option_shards<T: Clone>(shards: &[Vec<T>]) -> Vec<Option<Vec<T>>> {
    let mut result = Vec::with_capacity(shards.len());

    for v in shards.iter() {
        let inner: Vec<T> = v.clone();
        result.push(Some(inner));
    }
    result
}

fn option_shards_to_shards<T: Clone>(shards: &[Option<Vec<T>>]) -> Vec<Vec<T>> {
    let mut result = Vec::with_capacity(shards.len());

    for maybe_shard in shards {
        let shard = match maybe_shard {
            Some(x) => x,
            None => panic!("Missing shard"),
        };
        let inner: Vec<T> = shard.clone();
        result.push(inner);
    }
    result
}

#[test]
fn test_reconstruct_shards() {
    const SHARD_LEN: usize = 20;

    let r = ReedSolomon::new(8, 5).unwrap();

    let mut shards = make_random_shards(SHARD_LEN, 13);

    r.encode(&mut shards).unwrap();

    let master_copy = shards.clone();

    let mut shards = shards_to_option_shards(&shards);

    // Try to decode with all shards present
    r.reconstruct(&mut shards).unwrap();
    {
        let shards = option_shards_to_shards(&shards);
        assert_eq!(&shards, &master_copy);
        assert!(r.verify(&shards).unwrap());
        assert_eq!(&shards, &master_copy);
    }

    // Try to decode with 10 shards
    shards[0] = None;
    shards[2] = None;
    r.reconstruct(&mut shards).unwrap();
    {
        let shards = option_shards_to_shards(&shards);
        assert!(r.verify(&shards).unwrap());
        assert_eq!(&shards, &master_copy);
    }

    // Try to decode the same shards again to try to
    // trigger the usage of cached decode matrix
    shards[0] = None;
    shards[2] = None;
    r.reconstruct(&mut shards).unwrap();
    {
        let shards = option_shards_to_shards(&shards);
        assert!(r.verify(&shards).unwrap());
        assert_eq!(&shards, &master_copy);
    }

    // Try to decode with 6 data and 4 parity shards
    shards[0] = None;
    shards[2] = None;
    shards[12] = None;
    r.reconstruct(&mut shards).unwrap();
    {
        let shards = option_shards_to_shards(&shards);
        assert!(r.verify(&shards).unwrap());
        assert_eq!(&shards, &master_copy);
    }

    // Try to reconstruct data only
    shards[0] = None;
    shards[1] = None;
    shards[12] = None;
    // FIXME: r.reconstruct_data(&mut shards).unwrap();
    r.reconstruct(&mut shards).unwrap();
    {
        let data_shards = option_shards_to_shards(&shards[0..8]);
        assert_eq!(master_copy[0], data_shards[0]);
        assert_eq!(master_copy[1], data_shards[1]);
        assert_eq!(None, shards[12]);
    }

    // Try to decode with 7 data and 1 parity shards
    shards[0] = None;
    shards[1] = None;
    shards[9] = None;
    shards[10] = None;
    shards[11] = None;
    shards[12] = None;
    assert_eq!(
        r.reconstruct(&mut shards).unwrap_err(),
        Error::TooFewShardsPresent
    );
}
