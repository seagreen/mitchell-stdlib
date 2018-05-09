{-# language CPP                #-}
{-# language ExplicitNamespaces #-}

module Equality
  (
    Eq(..),
#ifdef CONTRAVARIANT
    Equivalence(..),
    defaultEquivalence,
    comparisonEquivalence,
#endif
    Eq1(..),
    eq1,
    Eq2(..),
    eq2,
    -- * Propositional equality
    (:~:)(..),
    (:~~:)(..),
    sym,
    trans,
    castWith,
    gcastWith,
    apply,
    inner,
    outer,
    TestEquality(..),
    type (==),
  ) where

import Data.Eq
import Data.Functor.Classes
#ifdef CONTRAVARIANT
import Data.Functor.Contravariant
#endif
import Data.Type.Equality
