# FabricPC — Julia port of Matthew Behrend's FabricPC (predictive-coding graph
# training framework). Layer 2 of the NGC Julia stack.
#
# See docs/DESIGN.md for the architecture + phased plan, and docs/decisions.md
# for cross-cutting decisions. Standalone package — no NGCSimLib/NGCLearn dep
# (FabricPC's Node/Edge/slot graph is a parallel abstraction; composition is a
# deferred Layer 3).

module FabricPC

const FABRICPC_VERSION = v"0.1.0"

# Phase B onward fills these in (core types → nodes → energy/inference/learning →
# assembly → training). Scaffold ships only the version constant.
export FABRICPC_VERSION

end # module FabricPC
