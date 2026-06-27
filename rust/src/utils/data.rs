//! Dataset helpers.

use std::path::Path;

use tch::{Device, IndexOp, Kind, Tensor};

use crate::error::{Result, TensorNetworkError};

/// Split a classification dataset class-by-class.
pub fn split_classification_dataset(
    data: &Tensor,
    targets: &Tensor,
    ratio: f64,
    shuffle: bool,
) -> (Tensor, Tensor, Tensor, Tensor) {
    assert!(
        matches!(targets.kind(), Kind::Int | Kind::Int64),
        "target must be an integer tensor"
    );
    assert!(1.0 > ratio && ratio > 0.0, "ratio must be between 0 and 1");
    assert_eq!(
        data.device(),
        targets.device(),
        "data and targets must be on the same device"
    );
    let device = targets.device();
    let num_classes = targets.max().int64_value(&[]) + 1;
    let num_samples = data.size()[0];
    assert_eq!(
        num_samples,
        targets.size()[0],
        "data and target must have the same number of samples"
    );
    let mut train_samples = Vec::new();
    let mut train_labels = Vec::new();
    let mut test_samples = Vec::new();
    let mut test_labels = Vec::new();
    for class_idx in 0..num_classes {
        let mask = targets.eq(class_idx);
        let mut class_samples = data.index(&[Some(&mask)]);
        let class_count = class_samples.size()[0];
        let train_count = ((class_count as f64) * (1.0 - ratio)) as i64;
        if shuffle {
            let permutation = Tensor::randperm(class_count, (Kind::Int64, device));
            class_samples = class_samples.index_select(0, &permutation);
        }
        let train = class_samples.i(..train_count);
        let test = class_samples.i(train_count..);
        train_labels.push(Tensor::full(
            [train.size()[0]],
            class_idx,
            (targets.kind(), device),
        ));
        test_labels.push(Tensor::full(
            [test.size()[0]],
            class_idx,
            (targets.kind(), device),
        ));
        train_samples.push(train);
        test_samples.push(test);
    }
    (
        Tensor::cat(&train_samples, 0),
        Tensor::cat(&train_labels, 0),
        Tensor::cat(&test_samples, 0),
        Tensor::cat(&test_labels, 0),
    )
}

/// Load cached MNIST data through `tch::vision::mnist`.
pub fn load_mnist_from_cache<P: AsRef<Path>>(
    cache_path: P,
) -> Result<tch::vision::dataset::Dataset> {
    if !cache_path.as_ref().exists() {
        return Err(TensorNetworkError::MissingArtifact(format!(
            "MNIST cache directory does not exist: {}",
            cache_path.as_ref().display()
        )));
    }
    Ok(tch::vision::mnist::load_dir(cache_path)?)
}

/// Move a dataset to a device.
pub fn dataset_to_device(
    dataset: tch::vision::dataset::Dataset,
    device: Device,
) -> tch::vision::dataset::Dataset {
    tch::vision::dataset::Dataset {
        train_images: dataset.train_images.to_device(device),
        train_labels: dataset.train_labels.to_device(device),
        test_images: dataset.test_images.to_device(device),
        test_labels: dataset.test_labels.to_device(device),
        labels: dataset.labels,
    }
}
