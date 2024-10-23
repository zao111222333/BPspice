use gspice_utils::expression::Expression;

use crate::node::Node;

pub struct Resistor {
    pub p: Node,
    pub n: Node,
    pub value: Expression,
}
