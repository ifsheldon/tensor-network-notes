//! Dataset helpers.

use std::path::Path;

use tch::{Device, IndexOp, Kind, Tensor};

use crate::error::{Result, TensorNetworkError};

/// Image dataset subset.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ImageSubset {
    /// Training split.
    Train,
    /// Test split.
    Test,
    /// Concatenated train and test splits.
    All,
}

/// Image preprocessing applied after selecting samples.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ImagePreprocess {
    /// Keep image values in the unit range produced by the IDX reader.
    UnitRange,
    /// Standardize unit-range image values by `(x - mean) / std`.
    Standardized {
        /// Dataset mean.
        mean: f64,
        /// Dataset standard deviation.
        std: f64,
    },
}

impl ImagePreprocess {
    /// Standard MNIST normalization used by the Python notebooks.
    pub fn mnist_standardized() -> Self {
        Self::Standardized {
            mean: 0.1307,
            std: 0.3081,
        }
    }
}

impl Default for ImagePreprocess {
    fn default() -> Self {
        Self::UnitRange
    }
}

/// Options for loading image tensors from cached classification datasets.
#[derive(Debug, Copy, Clone)]
pub struct ImageLoadOptions<'a> {
    /// Dataset split to load.
    pub subset: ImageSubset,
    /// Optional maximum number of images after filtering and shuffling.
    pub num: Option<i64>,
    /// Whether to shuffle after class filtering and before truncation.
    pub shuffle: bool,
    /// Image preprocessing.
    pub preprocess: ImagePreprocess,
    /// Optional class filter.
    pub classes: Option<&'a [i64]>,
    /// Device for returned tensors.
    pub device: Device,
}

impl Default for ImageLoadOptions<'_> {
    fn default() -> Self {
        Self {
            subset: ImageSubset::Train,
            num: None,
            shuffle: false,
            preprocess: ImagePreprocess::UnitRange,
            classes: None,
            device: Device::Cpu,
        }
    }
}

/// Loaded image tensors and labels.
#[derive(Debug)]
pub struct LoadedImages {
    /// Images shaped `(N, 1, 28, 28)`.
    pub images: Tensor,
    /// Labels shaped `(N)`.
    pub labels: Tensor,
}

/// Load the Iris dataset as tensors.
pub fn load_iris(force_single_precision: bool) -> (Tensor, Tensor) {
    let dataset = linfa_datasets::iris();
    let records = dataset.records();
    let data = Tensor::from_slice(records.as_slice().expect("Iris records are contiguous"))
        .reshape([records.nrows() as i64, records.ncols() as i64]);
    let data = if force_single_precision {
        data.to_kind(Kind::Float)
    } else {
        data
    };
    let targets = dataset
        .targets()
        .iter()
        .map(|target| *target as i64)
        .collect::<Vec<_>>();
    (data, Tensor::from_slice(&targets).to_kind(Kind::Int64))
}

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

/// Load selected MNIST images from a cache directory.
pub fn load_mnist_images_from_cache<P: AsRef<Path>>(
    cache_path: P,
    options: ImageLoadOptions<'_>,
) -> Result<LoadedImages> {
    let dataset = load_mnist_from_cache(cache_path)?;
    Ok(select_classification_images(&dataset, options))
}

/// Load cached Fashion-MNIST data through the MNIST IDX reader.
///
/// The cache must already contain uncompressed IDX files using the names expected by
/// `tch::vision::mnist::load_dir`.
pub fn load_fashion_mnist_from_cache<P: AsRef<Path>>(
    cache_path: P,
) -> Result<tch::vision::dataset::Dataset> {
    load_mnist_from_cache(cache_path)
}

/// Load selected Fashion-MNIST images from a cache directory.
pub fn load_fashion_mnist_images_from_cache<P: AsRef<Path>>(
    cache_path: P,
    options: ImageLoadOptions<'_>,
) -> Result<LoadedImages> {
    let dataset = load_fashion_mnist_from_cache(cache_path)?;
    Ok(select_classification_images(&dataset, options))
}

/// Select, preprocess, and move image tensors from a classification dataset.
pub fn select_classification_images(
    dataset: &tch::vision::dataset::Dataset,
    options: ImageLoadOptions<'_>,
) -> LoadedImages {
    if let Some(num) = options.num {
        assert!(num > 0, "num must be positive when provided");
    }
    validate_classes(options.classes, dataset.labels);
    let (mut images, mut labels) = match options.subset {
        ImageSubset::Train => (
            dataset.train_images.shallow_clone(),
            dataset.train_labels.shallow_clone(),
        ),
        ImageSubset::Test => (
            dataset.test_images.shallow_clone(),
            dataset.test_labels.shallow_clone(),
        ),
        ImageSubset::All => (
            Tensor::cat(&[&dataset.train_images, &dataset.test_images], 0),
            Tensor::cat(&[&dataset.train_labels, &dataset.test_labels], 0),
        ),
    };
    if let Some(classes) = options.classes {
        let mask = class_mask(&labels, classes);
        images = images.index(&[Some(&mask)]);
        labels = labels.index(&[Some(&mask)]);
    }
    if options.shuffle {
        let permutation = Tensor::randperm(labels.size()[0], (Kind::Int64, labels.device()));
        images = images.index_select(0, &permutation);
        labels = labels.index_select(0, &permutation);
    }
    if let Some(num) = options.num {
        let keep = num.min(labels.size()[0]);
        images = images.i(..keep);
        labels = labels.i(..keep);
    }
    images = apply_image_preprocess(&images, options.preprocess);
    images = image_tensor_to_nchw(&images).to_device(options.device);
    labels = labels.to_device(options.device);
    LoadedImages { images, labels }
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

fn validate_classes(classes: Option<&[i64]>, labels: i64) {
    if let Some(classes) = classes {
        assert!(!classes.is_empty(), "classes must be non-empty");
        let mut seen = Vec::with_capacity(classes.len());
        for &class_id in classes {
            assert!(
                0 <= class_id && class_id < labels,
                "class id must be in [0, labels)"
            );
            assert!(!seen.contains(&class_id), "classes must be unique");
            seen.push(class_id);
        }
    }
}

fn class_mask(labels: &Tensor, classes: &[i64]) -> Tensor {
    let mut mask = labels.eq(classes[0]);
    for &class_id in classes.iter().skip(1) {
        mask = mask.logical_or(&labels.eq(class_id));
    }
    mask
}

fn apply_image_preprocess(images: &Tensor, preprocess: ImagePreprocess) -> Tensor {
    match preprocess {
        ImagePreprocess::UnitRange => images.shallow_clone(),
        ImagePreprocess::Standardized { mean, std } => {
            assert!(std > 0.0, "standard deviation must be positive");
            (images - mean) / std
        }
    }
}

fn image_tensor_to_nchw(images: &Tensor) -> Tensor {
    let shape = images.size();
    match shape.as_slice() {
        [_, 784] => images.reshape([shape[0], 1, 28, 28]),
        [_, 1, 28, 28] => images.shallow_clone(),
        _ => panic!("images must have shape (N, 784) or (N, 1, 28, 28), got {shape:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_dataset() -> tch::vision::dataset::Dataset {
        let train_images = Tensor::arange(6 * 784, (Kind::Float, Device::Cpu)).reshape([6, 784])
            / (6 * 784) as f64;
        let train_labels = Tensor::from_slice(&[0_i64, 1, 2, 0, 1, 2]);
        let test_images = Tensor::arange_start(6 * 784, 10 * 784, (Kind::Float, Device::Cpu))
            .reshape([4, 784])
            / (10 * 784) as f64;
        let test_labels = Tensor::from_slice(&[0_i64, 1, 2, 1]);
        tch::vision::dataset::Dataset {
            train_images,
            train_labels,
            test_images,
            test_labels,
            labels: 3,
        }
    }

    #[test]
    fn load_iris_returns_expected_shapes_and_kinds() {
        let (data, targets) = load_iris(false);
        assert_eq!(data.size(), vec![150, 4]);
        assert_eq!(data.kind(), Kind::Double);
        assert_eq!(targets.size(), vec![150]);
        assert_eq!(targets.kind(), Kind::Int64);

        let (data_f32, _) = load_iris(true);
        assert_eq!(data_f32.kind(), Kind::Float);
    }

    #[test]
    fn select_classification_images_handles_subsets_classes_and_shapes() {
        let dataset = synthetic_dataset();
        let loaded = select_classification_images(
            &dataset,
            ImageLoadOptions {
                subset: ImageSubset::All,
                num: Some(3),
                classes: Some(&[1]),
                ..Default::default()
            },
        );
        assert_eq!(loaded.images.size(), vec![3, 1, 28, 28]);
        assert_eq!(loaded.labels.size(), vec![3]);
        assert!(loaded.labels.eq(1).all().int64_value(&[]) != 0);
    }

    #[test]
    fn select_classification_images_supports_test_subset() {
        let dataset = synthetic_dataset();
        let loaded = select_classification_images(
            &dataset,
            ImageLoadOptions {
                subset: ImageSubset::Test,
                num: None,
                ..Default::default()
            },
        );
        assert_eq!(loaded.images.size()[0], 4);
        assert_eq!(loaded.labels.size()[0], 4);
    }

    #[test]
    fn image_preprocess_unit_range_keeps_values_unchanged() {
        let dataset = synthetic_dataset();
        let loaded = select_classification_images(
            &dataset,
            ImageLoadOptions {
                subset: ImageSubset::Train,
                num: Some(1),
                preprocess: ImagePreprocess::UnitRange,
                ..Default::default()
            },
        );
        let expected = dataset.train_images.i((0, 0)).double_value(&[]);
        let actual = loaded.images.i((0, 0, 0, 0)).double_value(&[]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn image_preprocess_mnist_standardized_matches_python_constants() {
        let dataset = synthetic_dataset();
        let loaded = select_classification_images(
            &dataset,
            ImageLoadOptions {
                subset: ImageSubset::Train,
                num: Some(1),
                preprocess: ImagePreprocess::mnist_standardized(),
                ..Default::default()
            },
        );
        let expected = (0.0 - 0.1307) / 0.3081;
        let actual = loaded.images.i((0, 0, 0, 0)).double_value(&[]);
        assert!((actual - expected).abs() < 1e-6);
    }

    #[test]
    fn selected_images_move_to_requested_device() {
        let dataset = synthetic_dataset();
        let loaded = select_classification_images(
            &dataset,
            ImageLoadOptions {
                device: Device::Cpu,
                ..Default::default()
            },
        );
        assert_eq!(loaded.images.device(), Device::Cpu);
        assert_eq!(loaded.labels.device(), Device::Cpu);
        if tch::Cuda::is_available() {
            let loaded = select_classification_images(
                &dataset,
                ImageLoadOptions {
                    device: Device::Cuda(0),
                    ..Default::default()
                },
            );
            assert_eq!(loaded.images.device(), Device::Cuda(0));
            assert_eq!(loaded.labels.device(), Device::Cuda(0));
        }
    }

    #[test]
    fn missing_mnist_cache_returns_artifact_error() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let missing = tmp.path().join("missing");
        let err = load_mnist_from_cache(&missing).expect_err("missing cache should error");
        assert!(matches!(err, TensorNetworkError::MissingArtifact(_)));
    }
}
