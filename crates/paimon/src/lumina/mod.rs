// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

pub mod ffi;
pub mod reader;

use std::collections::HashMap;

pub const LUMINA_VECTOR_ANN_IDENTIFIER: &str = "lumina-vector-ann";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LuminaVectorMetric {
    L2,
    Cosine,
    InnerProduct,
}

impl LuminaVectorMetric {
    pub fn lumina_name(&self) -> &str {
        match self {
            LuminaVectorMetric::L2 => "l2",
            LuminaVectorMetric::Cosine => "cosine",
            LuminaVectorMetric::InnerProduct => "inner_product",
        }
    }

    pub fn from_string(name: &str) -> crate::Result<Self> {
        match name.to_uppercase().as_str() {
            "L2" => Ok(LuminaVectorMetric::L2),
            "COSINE" => Ok(LuminaVectorMetric::Cosine),
            "INNER_PRODUCT" => Ok(LuminaVectorMetric::InnerProduct),
            _ => Err(crate::Error::DataInvalid {
                message: format!("Unknown metric name: {}", name),
                source: None,
            }),
        }
    }

    pub fn from_lumina_name(lumina_name: &str) -> crate::Result<Self> {
        match lumina_name {
            "l2" => Ok(LuminaVectorMetric::L2),
            "cosine" => Ok(LuminaVectorMetric::Cosine),
            "inner_product" => Ok(LuminaVectorMetric::InnerProduct),
            _ => Err(crate::Error::DataInvalid {
                message: format!("Unknown lumina metric name: {}", lumina_name),
                source: None,
            }),
        }
    }
}

const LUMINA_PREFIX: &str = "lumina.";

const SEARCH_OPTIONS_DEFAULTS: &[(&str, &str)] = &[
    ("lumina.diskann.search.beam_width", "4"),
    ("lumina.search.parallel_number", "5"),
];

pub struct LuminaVectorIndexOptions {
    pub dimension: i32,
    pub metric: LuminaVectorMetric,
    pub index_type: String,
    lumina_options: HashMap<String, String>,
}

impl LuminaVectorIndexOptions {
    pub fn new(paimon_options: &HashMap<String, String>) -> crate::Result<Self> {
        let dimension_str = paimon_options
            .get("lumina.index.dimension")
            .map(|s| s.as_str())
            .unwrap_or("128");
        let dimension: i32 = dimension_str
            .parse()
            .map_err(|_| crate::Error::DataInvalid {
                message: format!("Invalid dimension: {}", dimension_str),
                source: None,
            })?;
        if dimension <= 0 {
            return Err(crate::Error::DataInvalid {
                message: format!(
                    "Invalid value for 'lumina.index.dimension': {}. Must be a positive integer.",
                    dimension
                ),
                source: None,
            });
        }

        let metric_str = paimon_options
            .get("lumina.distance.metric")
            .map(|s| s.as_str())
            .unwrap_or("inner_product");
        let metric = LuminaVectorMetric::from_lumina_name(metric_str)
            .or_else(|_| LuminaVectorMetric::from_string(metric_str))?;

        let index_type = paimon_options
            .get("lumina.index.type")
            .cloned()
            .unwrap_or_else(|| "diskann".to_string());

        let lumina_options = strip_lumina_options(paimon_options);

        Ok(Self {
            dimension,
            metric,
            index_type,
            lumina_options,
        })
    }

    pub fn to_lumina_options(&self) -> HashMap<String, String> {
        self.lumina_options.clone()
    }
}

pub fn strip_lumina_options(paimon_options: &HashMap<String, String>) -> HashMap<String, String> {
    let mut result = HashMap::new();

    for &(paimon_key, default_value) in SEARCH_OPTIONS_DEFAULTS {
        let native_key = &paimon_key[LUMINA_PREFIX.len()..];
        result.insert(native_key.to_string(), default_value.to_string());
    }

    for (key, value) in paimon_options {
        if let Some(native_key) = key.strip_prefix(LUMINA_PREFIX) {
            result.insert(native_key.to_string(), value.to_string());
        }
    }

    result
}

pub const KEY_DIMENSION: &str = "index.dimension";
pub const KEY_DISTANCE_METRIC: &str = "distance.metric";
pub const KEY_INDEX_TYPE: &str = "index.type";

pub struct LuminaIndexMeta {
    options: HashMap<String, String>,
}

impl LuminaIndexMeta {
    pub fn new(options: HashMap<String, String>) -> Self {
        Self { options }
    }

    pub fn options(&self) -> &HashMap<String, String> {
        &self.options
    }

    pub fn dim(&self) -> i32 {
        self.options
            .get(KEY_DIMENSION)
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(0)
    }

    pub fn distance_metric(&self) -> &str {
        self.options
            .get(KEY_DISTANCE_METRIC)
            .map(String::as_str)
            .unwrap_or("")
    }

    pub fn metric(&self) -> crate::Result<LuminaVectorMetric> {
        LuminaVectorMetric::from_lumina_name(self.distance_metric())
    }

    pub fn index_type(&self) -> &str {
        self.options
            .get(KEY_INDEX_TYPE)
            .map(String::as_str)
            .unwrap_or("diskann")
    }

    pub fn serialize(&self) -> crate::Result<Vec<u8>> {
        serde_json::to_vec(&self.options).map_err(|e| crate::Error::DataInvalid {
            message: format!("Failed to serialize LuminaIndexMeta: {}", e),
            source: None,
        })
    }

    pub fn deserialize(data: &[u8]) -> crate::Result<Self> {
        let options: HashMap<String, String> =
            serde_json::from_slice(data).map_err(|e| crate::Error::DataInvalid {
                message: format!("Failed to deserialize LuminaIndexMeta: {}", e),
                source: None,
            })?;
        if !options.contains_key(KEY_DIMENSION) {
            return Err(crate::Error::DataInvalid {
                message: format!(
                    "Missing required key in Lumina index metadata: {}",
                    KEY_DIMENSION
                ),
                source: None,
            });
        }
        if !options.contains_key(KEY_DISTANCE_METRIC) {
            return Err(crate::Error::DataInvalid {
                message: format!(
                    "Missing required key in Lumina index metadata: {}",
                    KEY_DISTANCE_METRIC
                ),
                source: None,
            });
        }
        Ok(Self { options })
    }
}
