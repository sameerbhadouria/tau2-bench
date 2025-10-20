# Policy

## Book flights

- verify booking circular loop
- For A -> B -> C -> A -> B, verify the 2nd B is not before 1st B. Is this considered multi city or return flight

## Modify flights

- Verify change flight rules
- Verify change cabin refund does apply to travel certificate
- verify modify passengers but not num passengers
- Verify single card for payment/refund on modify

## Cancelation

- verify insurance rules

## Compensation

- verify the compensation is for silver/gold only if asked by user and following 2 reasosn
  - for cancelled flights - limited to $100 x # passenger
  - for delayed flights - limited to $50 x # passenger

## Summary Sonnet eval

The task suite provides excellent coverage of adversarial scenarios (users lying, manipulating, pressuring) and core policy enforcement. However, it could be strengthened by adding edge cases for:
Partially flown reservations
Airline-initiated cancellations
Maximum capacity limits (passengers, gift cards)
Comprehensive baggage benefit testing across all membership tiers
Flight availability status restrictions
The policy document is generally well-written but would benefit from clarifications around time calculations, insurance claim processes, and flight segment pricing rules.
