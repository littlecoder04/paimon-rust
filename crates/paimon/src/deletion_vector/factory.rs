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

#![allow(dead_code)]

use crate::deletion_vector::core::DeletionVector;
use crate::io::{FileIO, FileRead};
use crate::Result;
use std::collections::HashMap;
use std::sync::Arc;

/// Factory for creating DeletionVector instances from files and metadata.
///
/// Corresponds to Java's [DeletionVector.Factory](https://github.com/apache/paimon/blob/master/paimon-core/src/main/java/org/apache/paimon/deletionvectors/DeletionVector.java)
/// (create(fileName) -> Optional<DeletionVector>). Can be built from split-level deletion files
/// ([create_from_deletion_files]) or from index manifest entries ([create]).
pub struct DeletionVectorFactory {
    /// Map from data file name to its deletion vector
    deletion_vectors: HashMap<String, Arc<DeletionVector>>,
}

impl DeletionVectorFactory {
    /// Create a DeletionVectorFactory from data file names and their optional deletion files.
    /// Same as Java's `DeletionVector.factory(fileIO, files, deletionFiles)`: for each file that
    /// has a DeletionFile, reads path/offset/length and loads the DV.
    pub async fn new(
        file_io: &FileIO,
        entries: Vec<(String, Option<crate::DeletionFile>)>,
    ) -> Result<Self> {
        let mut deletion_vectors = HashMap::new();
        for (data_file_name, opt_df) in entries {
            let df = match &opt_df {
                Some(d) => d,
                _ => continue,
            };
            let dv = Self::read(file_io, df).await?;
            deletion_vectors.insert(data_file_name, Arc::new(dv));
        }
        Ok(DeletionVectorFactory { deletion_vectors })
    }

    /// Get the deletion vector for a specific data file
    pub fn get_deletion_vector(&self, data_file_name: &str) -> Option<Arc<DeletionVector>> {
        self.deletion_vectors.get(data_file_name).cloned()
    }

    /// Read a single DeletionVector from storage using DeletionFile (path/offset/length).
    /// Same as Java's DeletionVector.read(FileIO, DeletionFile).
    async fn read(file_io: &FileIO, df: &crate::DeletionFile) -> Result<DeletionVector> {
        let input = file_io.new_input(df.path())?;
        let reader = input.reader().await?;
        let offset = df.offset() as u64;
        let len = df.length() as u64;
        let bytes = reader.read(offset..offset.saturating_add(len)).await?;
        DeletionVector::read_from_bytes(&bytes, Some(len))
    }
}
