{-# language CPP #-}

#ifdef CONTAINERS

module LazyIntMap
  ( module Data.IntMap.Lazy
  ) where

import Data.IntMap.Lazy

#else

module LazyIntMap where

#endif
