//! A solid wrist socket with native inventory status art in its recessed bed.
use crate::ui::Rect;
use cgmath::Vector2;

pub(crate) const INVENTORY_SIZE: Vector2<f32> = Vector2::new(636.0, 121.0);

pub(crate) fn inventory_well(slot: usize) -> Rect {
    Rect::new(563.0 + slot as f32 * 36.0, 84.0, 34.0, 34.0)
}

#[cfg(test)]
mod tests {}
