//! `org.apache.lucene.tests.geo`: the random point, box and circle
//! generators and the point-query base that checks an index of random
//! points against a brute-force model.

pub mod base_geo_point;
pub mod base_xy_point;
pub mod geo_test_util;
mod point_index;
pub mod shape_test_util;
