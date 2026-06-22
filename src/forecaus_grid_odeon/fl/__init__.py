"""Federated learning layer for Challenge 4.

ODEON mandates FL "natively end-to-end" and provides the Federated Learning
Engine; the non-overlap rule means we CONSUME it rather than build one. This
package holds (a) a transparent FedAvg reference over interpretable model
parameters, usable for a pre-submission prototype across SS-like nodes, and
(b) a stub client adapter to the ODEON FL Engine (API pending, help-desk Q1).
See ODEON_Challenge4_FL_Architecture.docx.
"""
from .federated import fedavg

__all__ = ["fedavg"]
