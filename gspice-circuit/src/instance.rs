use gspice_utils::Expression;

use crate::node::Node;

pub struct Resistor {
    pub p: Node,
    pub n: Node,
    pub value: Expression,
}
