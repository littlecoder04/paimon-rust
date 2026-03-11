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

//! Table source types: DataSplit, Plan, and related structs.
//!
//! Reference: [org.apache.paimon.table.source](https://github.com/apache/paimon/blob/master/paimon-core/src/main/java/org/apache/paimon/table/source/).

#![allow(dead_code)]

use crate::spec::{BinaryRow, DataFileMeta};

// ======================= DataSplit ===============================

/// Input split for reading: partition + bucket + list of data files.
///
/// Reference: [org.apache.paimon.table.source.DataSplit](https://github.com/apache/paimon/blob/release-1.3/paimon-core/src/main/java/org/apache/paimon/table/source/DataSplit.java)
#[derive(Debug)]
pub struct DataSplit {
    snapshot_id: i64,
    partition: BinaryRow,
    bucket: i32,
    bucket_path: String,
    total_buckets: i32,
    data_files: Vec<DataFileMeta>,
}

impl DataSplit {
    pub fn snapshot_id(&self) -> i64 {
        self.snapshot_id
    }
    pub fn partition(&self) -> &BinaryRow {
        &self.partition
    }
    pub fn bucket(&self) -> i32 {
        self.bucket
    }
    pub fn bucket_path(&self) -> &str {
        &self.bucket_path
    }
    pub fn total_buckets(&self) -> i32 {
        self.total_buckets
    }

    pub fn data_files(&self) -> &[DataFileMeta] {
        &self.data_files
    }

    /// Iterate over each data file in this split, yielding `(path, &DataFileMeta)`.
    /// Use this to read each data file one by one (e.g. in ArrowReader).
    pub fn data_file_entries(&self) -> impl Iterator<Item = (String, &DataFileMeta)> + '_ {
        let base = self.bucket_path.trim_end_matches('/');
        // todo: consider partition table
        // todo: consider external path
        self.data_files.iter().map(move |file| {
            let path = format!("{}/{}", base, file.file_name);
            (path, file)
        })
    }

    /// Total row count of all data files in this split.
    pub fn row_count(&self) -> i64 {
        self.data_files.iter().map(|f| f.row_count).sum()
    }

    pub fn builder() -> DataSplitBuilder {
        DataSplitBuilder::new()
    }
}

/// Builder for [DataSplit].
///
/// Reference: [DataSplit.Builder](https://github.com/apache/paimon/blob/release-1.3/paimon-core/src/main/java/org/apache/paimon/table/source/DataSplit.java)
#[derive(Debug)]
pub struct DataSplitBuilder {
    snapshot_id: i64,
    partition: Option<BinaryRow>,
    bucket: i32,
    bucket_path: Option<String>,
    total_buckets: i32,
    data_files: Option<Vec<DataFileMeta>>,
}

impl DataSplitBuilder {
    pub fn new() -> Self {
        Self {
            snapshot_id: -1,
            partition: None,
            bucket: -1,
            bucket_path: None,
            total_buckets: -1,
            data_files: None,
        }
    }

    pub fn with_snapshot(mut self, snapshot_id: i64) -> Self {
        self.snapshot_id = snapshot_id;
        self
    }
    pub fn with_partition(mut self, partition: BinaryRow) -> Self {
        self.partition = Some(partition);
        self
    }
    pub fn with_bucket(mut self, bucket: i32) -> Self {
        self.bucket = bucket;
        self
    }
    pub fn with_bucket_path(mut self, bucket_path: String) -> Self {
        self.bucket_path = Some(bucket_path);
        self
    }
    pub fn with_total_buckets(mut self, total_buckets: i32) -> Self {
        self.total_buckets = total_buckets;
        self
    }
    pub fn with_data_files(mut self, data_files: Vec<DataFileMeta>) -> Self {
        self.data_files = Some(data_files);
        self
    }

    pub fn build(self) -> crate::Result<DataSplit> {
        if self.snapshot_id == -1 {
            return Err(crate::Error::UnexpectedError {
                message: "DataSplit requires snapshot_id != -1".to_string(),
                source: None,
            });
        }
        let partition = self
            .partition
            .ok_or_else(|| crate::Error::UnexpectedError {
                message: "DataSplit requires partition".to_string(),
                source: None,
            })?;
        let bucket_path = self
            .bucket_path
            .ok_or_else(|| crate::Error::UnexpectedError {
                message: "DataSplit requires bucket_path".to_string(),
                source: None,
            })?;
        let data_files = self
            .data_files
            .ok_or_else(|| crate::Error::UnexpectedError {
                message: "DataSplit requires data_files".to_string(),
                source: None,
            })?;
        if self.bucket == -1 {
            return Err(crate::Error::UnexpectedError {
                message: "DataSplit requires bucket != -1".to_string(),
                source: None,
            });
        }
        Ok(DataSplit {
            snapshot_id: self.snapshot_id,
            partition,
            bucket: self.bucket,
            bucket_path,
            total_buckets: self.total_buckets,
            data_files,
        })
    }
}

impl Default for DataSplitBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ======================= Plan ===============================

/// Read plan: list of splits.
///
/// Reference: [org.apache.paimon.table.source.PlanImpl](https://github.com/apache/paimon/blob/release-1.3/paimon-core/src/main/java/org/apache/paimon/table/source/PlanImpl.java)
#[derive(Debug)]
pub struct Plan {
    splits: Vec<DataSplit>,
}

impl Plan {
    pub fn new(splits: Vec<DataSplit>) -> Self {
        Self { splits }
    }
    pub fn splits(&self) -> &[DataSplit] {
        &self.splits
    }
}
