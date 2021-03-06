{-# language CPP #-}

module Json.Decode
  (
    -- * Decoding
    FromJSON(..),
#ifdef DEP_GENERIC_AESON
    gparseJson,
#endif
    FromJSONKey(..),
    FromJSONKeyFunction(..),
    fromJSON,
    decode,
    decode',
    decodeStrict,
    decodeStrict',
    eitherDecode,
    eitherDecode',
    eitherDecodeStrict,
    eitherDecodeStrict',
    decodeWith,
    decodeStrictWith,
    eitherDecodeWith,
    eitherDecodeStrictWith,
    withObject,
    withText,
    withArray,
    withScientific,
    withBool,
    withEmbeddedJSON,
    (.:),
    (.:?),
    (.:!),
    (.!=),
    parseField,
    parseFieldMaybe,
    parseFieldMaybe',
    explicitParseField,
    explicitParseFieldMaybe,
    explicitParseFieldMaybe',
    Parser,
    Result(..),
    parse,
    parseMaybe,
    parseEither,
    iparse,
    json,
    json',
    value,
    value',
    jstring,
    scientific,
    -- * Re-exports
    module Json,
  ) where

import Json

import Data.Aeson
import Data.Aeson.Internal
import Data.Aeson.Parser
import Data.Aeson.Types
#ifdef DEP_GENERIC_AESON
import Generics.Generic.Aeson (gparseJson)
#endif
