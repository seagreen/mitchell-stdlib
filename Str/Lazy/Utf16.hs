{-# language CPP #-}

module Str.Lazy.Utf16
  ( module Data.Text.Lazy
  , module Data.Text.Lazy.Encoding
  ) where

import Data.Text.Lazy
import Data.Text.Lazy.Encoding

#ifdef MITCHELL
type Utf16
  = Text
#endif