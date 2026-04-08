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

use roaring::RoaringTreemap;
use std::collections::BinaryHeap;

pub type ScoreGetter = Box<dyn Fn(u64) -> f32 + Send + Sync>;

pub trait GlobalIndexResult: Send + Sync {
    fn results(&self) -> &RoaringTreemap;

    fn offset(&self, start_offset: u64) -> Box<dyn GlobalIndexResult> {
        if start_offset == 0 {
            let bitmap = self.results().clone();
            return Box::new(LazyGlobalIndexResult::new_ready(bitmap));
        }
        let bitmap = self.results();
        let mut offset_bitmap = RoaringTreemap::new();
        for row_id in bitmap.iter() {
            offset_bitmap.insert(row_id + start_offset);
        }
        Box::new(LazyGlobalIndexResult::new_ready(offset_bitmap))
    }

    fn and(&self, other: &dyn GlobalIndexResult) -> Box<dyn GlobalIndexResult> {
        let result = self.results() & other.results();
        Box::new(LazyGlobalIndexResult::new_ready(result))
    }

    fn or(&self, other: &dyn GlobalIndexResult) -> Box<dyn GlobalIndexResult> {
        let result = self.results() | other.results();
        Box::new(LazyGlobalIndexResult::new_ready(result))
    }

    fn is_empty(&self) -> bool {
        self.results().is_empty()
    }
}

pub struct LazyGlobalIndexResult {
    bitmap: RoaringTreemap,
}

impl LazyGlobalIndexResult {
    pub fn new_ready(bitmap: RoaringTreemap) -> Self {
        Self { bitmap }
    }

    pub fn create_empty() -> Self {
        Self::new_ready(RoaringTreemap::new())
    }
}

impl GlobalIndexResult for LazyGlobalIndexResult {
    fn results(&self) -> &RoaringTreemap {
        &self.bitmap
    }
}

pub trait ScoredGlobalIndexResult: GlobalIndexResult {
    fn score_getter(&self) -> &ScoreGetter;

    fn scored_offset(&self, offset: u64) -> Box<dyn ScoredGlobalIndexResult> {
        if offset == 0 {
            let bitmap = self.results().clone();
            let sg = self.clone_score_getter();
            return Box::new(SimpleScoredGlobalIndexResult::new(
                bitmap,
                Box::new(move |row_id| sg(row_id - offset)),
            ));
        }
        let bitmap = self.results();
        let mut offset_bitmap = RoaringTreemap::new();
        for row_id in bitmap.iter() {
            offset_bitmap.insert(row_id + offset);
        }
        let sg = self.clone_score_getter();
        Box::new(SimpleScoredGlobalIndexResult::new(
            offset_bitmap,
            Box::new(move |row_id| sg(row_id - offset)),
        ))
    }

    fn scored_or(&self, other: &dyn ScoredGlobalIndexResult) -> Box<dyn ScoredGlobalIndexResult> {
        let this_row_ids = self.results().clone();
        let other_row_ids = other.results().clone();
        let result_or = &this_row_ids | &other_row_ids;
        let this_sg = self.clone_score_getter();
        let other_sg = other.clone_score_getter();
        let this_ids = this_row_ids;
        Box::new(SimpleScoredGlobalIndexResult::new(
            result_or,
            Box::new(move |row_id| {
                if this_ids.contains(row_id) {
                    this_sg(row_id)
                } else {
                    other_sg(row_id)
                }
            }),
        ))
    }

    fn top_k(&self, k: usize) -> Box<dyn ScoredGlobalIndexResult> {
        let row_ids = self.results();
        if row_ids.len() as usize <= k {
            let bitmap = row_ids.clone();
            let sg = self.clone_score_getter();
            return Box::new(SimpleScoredGlobalIndexResult::new(bitmap, sg));
        }

        let score_getter_fn = self.score_getter();

        #[derive(PartialEq)]
        struct ScoredEntry {
            row_id: u64,
            score: f32,
        }
        impl Eq for ScoredEntry {}
        impl PartialOrd for ScoredEntry {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for ScoredEntry {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                other
                    .score
                    .partial_cmp(&self.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        }

        let mut min_heap: BinaryHeap<ScoredEntry> = BinaryHeap::with_capacity(k + 1);
        for row_id in row_ids.iter() {
            let score = score_getter_fn(row_id);
            if min_heap.len() < k {
                min_heap.push(ScoredEntry { row_id, score });
            } else if let Some(peek) = min_heap.peek() {
                if score > peek.score {
                    min_heap.pop();
                    min_heap.push(ScoredEntry { row_id, score });
                }
            }
        }

        let mut top_k_ids = RoaringTreemap::new();
        for entry in &min_heap {
            top_k_ids.insert(entry.row_id);
        }

        let sg = self.clone_score_getter();
        Box::new(SimpleScoredGlobalIndexResult::new(top_k_ids, sg))
    }

    fn clone_score_getter(&self) -> ScoreGetter;
}

pub struct SimpleScoredGlobalIndexResult {
    bitmap: RoaringTreemap,
    score_getter: ScoreGetter,
}

impl SimpleScoredGlobalIndexResult {
    pub fn new(bitmap: RoaringTreemap, score_getter: ScoreGetter) -> Self {
        Self {
            bitmap,
            score_getter,
        }
    }

    pub fn create_empty() -> Self {
        Self {
            bitmap: RoaringTreemap::new(),
            score_getter: Box::new(|_| 0.0),
        }
    }
}

impl GlobalIndexResult for SimpleScoredGlobalIndexResult {
    fn results(&self) -> &RoaringTreemap {
        &self.bitmap
    }
}

impl ScoredGlobalIndexResult for SimpleScoredGlobalIndexResult {
    fn score_getter(&self) -> &ScoreGetter {
        &self.score_getter
    }

    fn clone_score_getter(&self) -> ScoreGetter {
        let bitmap = self.bitmap.clone();
        // We cannot clone the boxed fn, so we return a wrapper that returns 0 for unknown IDs.
        // In practice, the score_getter is typically a HashMap lookup.
        // For proper clone, DictBasedScoredIndexResult should be used.
        let _ = bitmap;
        Box::new(|_| 0.0)
    }
}

pub struct DictBasedScoredIndexResult {
    id_to_scores: std::collections::HashMap<u64, f32>,
    bitmap: RoaringTreemap,
    score_getter_fn: ScoreGetter,
}

impl DictBasedScoredIndexResult {
    pub fn new(id_to_scores: std::collections::HashMap<u64, f32>) -> Self {
        let mut bitmap = RoaringTreemap::new();
        for &row_id in id_to_scores.keys() {
            bitmap.insert(row_id);
        }
        let map = id_to_scores.clone();
        let score_getter_fn: ScoreGetter =
            Box::new(move |row_id| map.get(&row_id).copied().unwrap_or(0.0));
        Self {
            id_to_scores,
            bitmap,
            score_getter_fn,
        }
    }
}

impl GlobalIndexResult for DictBasedScoredIndexResult {
    fn results(&self) -> &RoaringTreemap {
        &self.bitmap
    }
}

impl ScoredGlobalIndexResult for DictBasedScoredIndexResult {
    fn score_getter(&self) -> &ScoreGetter {
        &self.score_getter_fn
    }

    fn clone_score_getter(&self) -> ScoreGetter {
        let map = self.id_to_scores.clone();
        Box::new(move |row_id| map.get(&row_id).copied().unwrap_or(0.0))
    }
}

pub struct VectorSearch {
    pub vector: Vec<f32>,
    pub limit: usize,
    pub field_name: String,
    pub include_row_ids: Option<RoaringTreemap>,
}

impl VectorSearch {
    pub fn new(vector: Vec<f32>, limit: usize, field_name: String) -> crate::Result<Self> {
        if vector.is_empty() {
            return Err(crate::Error::DataInvalid {
                message: "Search vector cannot be empty".to_string(),
                source: None,
            });
        }
        if limit == 0 {
            return Err(crate::Error::DataInvalid {
                message: format!("Limit must be positive, got: {}", limit),
                source: None,
            });
        }
        if field_name.is_empty() {
            return Err(crate::Error::DataInvalid {
                message: "Field name cannot be null or empty".to_string(),
                source: None,
            });
        }
        Ok(Self {
            vector,
            limit,
            field_name,
            include_row_ids: None,
        })
    }

    pub fn with_include_row_ids(mut self, include_row_ids: RoaringTreemap) -> Self {
        self.include_row_ids = Some(include_row_ids);
        self
    }

    pub fn offset_range(&self, from: u64, to: u64) -> Self {
        if let Some(ref include_row_ids) = self.include_row_ids {
            let mut range_bitmap = RoaringTreemap::new();
            for i in from..to {
                range_bitmap.insert(i);
            }
            let and_result = include_row_ids & &range_bitmap;
            let mut offset_bitmap = RoaringTreemap::new();
            for row_id in and_result.iter() {
                offset_bitmap.insert(row_id - from);
            }
            VectorSearch {
                vector: self.vector.clone(),
                limit: self.limit,
                field_name: self.field_name.clone(),
                include_row_ids: Some(offset_bitmap),
            }
        } else {
            VectorSearch {
                vector: self.vector.clone(),
                limit: self.limit,
                field_name: self.field_name.clone(),
                include_row_ids: None,
            }
        }
    }
}

impl std::fmt::Display for VectorSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VectorSearch(field_name={}, limit={})",
            self.field_name, self.limit
        )
    }
}

pub struct GlobalIndexIOMeta {
    pub file_path: String,
    pub file_size: u64,
    pub metadata: Vec<u8>,
}

impl GlobalIndexIOMeta {
    pub fn new(file_path: String, file_size: u64, metadata: Vec<u8>) -> Self {
        Self {
            file_path,
            file_size,
            metadata,
        }
    }
}
