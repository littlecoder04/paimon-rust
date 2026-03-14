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

use std::ffi::{c_char, c_void};

use paimon::catalog::Identifier;
use paimon::{Catalog, FileSystemCatalog};

use crate::error::{check_non_null, paimon_error, validate_cstr};
use crate::result::{paimon_result_catalog_new, paimon_result_get_table};
use crate::runtime;
use crate::types::{paimon_catalog, paimon_table};

/// Create a new FileSystemCatalog.
///
/// # Safety
/// `warehouse` must be a valid null-terminated C string, or null (returns error).
#[no_mangle]
pub unsafe extern "C" fn paimon_catalog_new(warehouse: *const c_char) -> paimon_result_catalog_new {
    let warehouse_str = match validate_cstr(warehouse, "warehouse") {
        Ok(s) => s,
        Err(e) => {
            return paimon_result_catalog_new {
                catalog: std::ptr::null_mut(),
                error: e,
            }
        }
    };
    match FileSystemCatalog::new(warehouse_str) {
        Ok(catalog) => {
            let wrapper = Box::new(paimon_catalog {
                inner: Box::into_raw(Box::new(catalog)) as *mut c_void,
            });
            paimon_result_catalog_new {
                catalog: Box::into_raw(wrapper),
                error: std::ptr::null_mut(),
            }
        }
        Err(e) => paimon_result_catalog_new {
            catalog: std::ptr::null_mut(),
            error: paimon_error::from_paimon(e),
        },
    }
}

/// Free a paimon_catalog.
///
/// # Safety
/// Only call with a catalog returned from `paimon_catalog_new`.
#[no_mangle]
pub unsafe extern "C" fn paimon_catalog_free(catalog: *mut paimon_catalog) {
    if !catalog.is_null() {
        let c = Box::from_raw(catalog);
        if !c.inner.is_null() {
            drop(Box::from_raw(c.inner as *mut FileSystemCatalog));
        }
    }
}

/// Get a table from the catalog.
///
/// # Safety
/// `catalog` and `identifier` must be valid pointers from previous paimon C calls, or null (returns error).
#[no_mangle]
pub unsafe extern "C" fn paimon_catalog_get_table(
    catalog: *const paimon_catalog,
    identifier: *const crate::types::paimon_identifier,
) -> paimon_result_get_table {
    if let Err(e) = check_non_null(catalog, "catalog") {
        return paimon_result_get_table {
            table: std::ptr::null_mut(),
            error: e,
        };
    }
    if let Err(e) = check_non_null(identifier, "identifier") {
        return paimon_result_get_table {
            table: std::ptr::null_mut(),
            error: e,
        };
    }

    let catalog_ref = &*((*catalog).inner as *const FileSystemCatalog);
    let identifier_ref = &*((*identifier).inner as *const Identifier);

    match runtime().block_on(catalog_ref.get_table(identifier_ref)) {
        Ok(table) => {
            let wrapper = Box::new(paimon_table {
                inner: Box::into_raw(Box::new(table)) as *mut c_void,
            });
            paimon_result_get_table {
                table: Box::into_raw(wrapper),
                error: std::ptr::null_mut(),
            }
        }
        Err(e) => paimon_result_get_table {
            table: std::ptr::null_mut(),
            error: paimon_error::from_paimon(e),
        },
    }
}
