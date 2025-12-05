use thiserror::Error as ThisError;

#[derive(ThisError, Debug)]
pub enum Error {
    #[error("Incorrect shard size")]
    IncorrectShardSize,
    #[error("Too few data shards")]
    TooFewDataShards,
    #[error("Too few parity shards")]
    TooFewParityShards,
    #[error("Too many shards")]
    TooManyShards,
    #[error("Empty shard")]
    EmptyShard,
    #[error("Too few shards present")]
    TooFewShardsPresent,
}
