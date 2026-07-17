use safetensors::tensor::Dtype;
use std::borrow::Cow;

/// An owned tensor, mirroring safetensors data without borrow lifetime.
#[derive(Debug, Clone)]
pub struct TensorData {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl TensorData {
    pub fn new(dtype: Dtype, shape: Vec<usize>, data: Vec<u8>) -> Self {
        Self { dtype, shape, data }
    }
}

impl safetensors::tensor::View for &TensorData {
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}
