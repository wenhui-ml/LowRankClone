from datasets import IterableDataset

def gen():
    for i in range(10):
        yield {"x": i}

dataset = IterableDataset.from_generator(gen)
ds0 = dataset.shard(num_shards=2, index=0)
ds1 = dataset.shard(num_shards=2, index=1)

print("Shard 0:", [x for x in ds0])
print("Shard 1:", [x for x in ds1])
