from datasets import load_dataset, DatasetDict, concatenate_datasets


if __name__ == "__main__":

    iter_1_train = load_dataset("GitBag/llama3-ultrafeedback-armo-1024-20k-iter_1_1723066371_harvard", split="train_prefs")
    test_data = load_dataset("GitBag/llama3-ultrafeedback-armo-1024-test_harvard", split="test_prefs")

    iter_0_train = load_dataset("GitBag/llama3-ultrafeedback-armo-1024_harvard", split="train_prefs")
    iter_0_train = iter_0_train.select(range(20000))

    train_data = concatenate_datasets([iter_0_train, iter_1_train])

    data = DatasetDict({
        "train_prefs" : train_data,
        "test_prefs"  : test_data,
    })
    data.push_to_hub("GitBag/llama3-ultrafeedback-armo-1024-20k-base-20k-1723066371")

    