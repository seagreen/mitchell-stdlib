{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE NoImplicitPrelude     #-}
{-# LANGUAGE PatternSynonyms       #-}
{-# LANGUAGE TypeSynonymInstances  #-}


module Mitchell.Prelude
  (
    -- async
    module Control.Concurrent.Async
    -- base
  , Control.Applicative.Applicative(..)
  , Control.Applicative.Const(..)
  , lift2
  , lift3
  , lift4
  , lift5
  , Control.Applicative.Alternative(..)
  , Control.Applicative.optional
  , module Control.Concurrent
  , Control.Exception.SomeException
  , Control.Exception.Exception(..)
  , Control.Exception.catch
  , Control.Exception.catchJust
  , Control.Exception.try
  , Control.Exception.tryJust
  , Control.Exception.evaluate
  , Control.Exception.assert
  , Control.Exception.bracket
  , Control.Exception.bracket_
  , Control.Exception.bracketOnError
  , Control.Exception.finally
  , Control.Exception.onException
  , Control.Monad.Monad((>>=), (>>), return)
  , Control.Monad.MonadPlus(..)
  , (Control.Monad.=<<)
  , (Control.Monad.>=>)
  , (Control.Monad.<=<)
  , Control.Monad.forever
  , Control.Monad.join
  , Control.Monad.mfilter
  , Control.Monad.filterM
  , Control.Monad.mapAndUnzipM
  , Control.Monad.zipWithM
  , Control.Monad.zipWithM_
  , Control.Monad.foldM
  , Control.Monad.foldM_
  , Control.Monad.replicateM
  , Control.Monad.replicateM_
  , Control.Monad.guard
  , Control.Monad.when
  , Control.Monad.unless
  , module Data.Bifunctor
  , module Data.Bool
  , module Data.Char
  , module Data.Coerce
  , module Data.Either
  , module Data.Eq
  , module Data.Foldable
  , Data.Function.const
  , (Data.Function..)
  , Data.Function.flip
  , (Data.Function.$)
  , (Data.Function.&)
  , Data.Function.fix
  , Data.Function.on
  , Data.Functor.Functor((<$))
  , (Data.Functor.$>)
  , (Data.Functor.<$>)
  , Data.Functor.void
  , identity
  , map
  , module Data.Functor.Identity
  , module Data.Int
  , module Data.IORef
  , Data.List.break
  , Data.List.cycle
  , Data.List.drop
  , Data.List.dropWhile
  , Data.List.inits
  , Data.List.isInfixOf
  , Data.List.isPrefixOf
  , Data.List.isSubsequenceOf
  , Data.List.isSuffixOf
  , Data.List.iterate
  , Data.List.filter
  , Data.List.group
  , Data.List.groupBy
  , Data.List.permutations
  , Data.List.repeat
  , Data.List.replicate
  , Data.List.reverse
  , Data.List.scanl
  , Data.List.scanl'
  , Data.List.scanl1
  , Data.List.scanr
  , Data.List.scanr1
  , Data.List.sort
  , Data.List.sortBy
  , Data.List.sortOn
  , Data.List.span
  , Data.List.splitAt
  , Data.List.subsequences
  , Data.List.tails
  , Data.List.take
  , Data.List.takeWhile
  , Data.List.transpose
  , Data.List.uncons
  , Data.List.unfoldr
  , Data.List.unzip
  , Data.List.unzip3
  , Data.List.unzip4
  , Data.List.unzip5
  , Data.List.unzip6
  , Data.List.unzip7
  , Data.List.zip
  , Data.List.zip3
  , Data.List.zip4
  , Data.List.zip5
  , Data.List.zip6
  , Data.List.zip7
  , Data.List.zipWith
  , Data.List.zipWith3
  , Data.List.zipWith4
  , Data.List.zipWith5
  , Data.List.zipWith6
  , Data.List.zipWith7
  , (!!)
  , head
  , init
  , last
  , tail
  , unsafeHead
  , unsafeTail
  , unsafeInit
  , unsafeLast
  , unsafeIndex
  , Data.Maybe.Maybe(..)
  , Data.Maybe.maybe
  , Data.Maybe.isJust
  , Data.Maybe.isNothing
  , Data.Maybe.fromMaybe
  , Data.Maybe.listToMaybe
  , Data.Maybe.maybeToList
  , Data.Maybe.catMaybes
  , Data.Maybe.mapMaybe
  , unsafeFromJust
  , Data.Monoid.Monoid
  , Data.Monoid.mconcat
  , zero
  , (++)
  , Data.Ord.Ord(..)
  , Data.Ord.Ordering(..)
  , Data.Ord.comparing
  , module Data.Proxy
  , module Data.Traversable
  , module Data.Tuple
  , Data.Typeable.Typeable
  , Data.Typeable.TypeRep
  , Data.Typeable.typeRep
  , trace
  , traceM
  , traceIO
  , traceShow
  , traceShowM
  , undefined
  , error
  , (GHC.Base.$!)
  , module GHC.Enum
  , GHC.Exts.Constraint
  , GHC.Exts.IsList(fromList)
  , GHC.Generics.Generic
  , module GHC.Float
  , module GHC.Num
  , GHC.Prim.seq
  , module GHC.Real
  , GHC.Show.Show
  , System.IO.IO
  , show
  , putStr
  , putStrLn
  , print
  , Text.Printf.printf
  , Text.Printf.hPrintf
    -- bytestring
  , Data.ByteString.ByteString
  , LByteString
    -- containers
  , Data.IntMap.Strict.IntMap
  , LIntMap
  , Data.IntSet.IntSet
  , Data.Map.Strict.Map
  , LMap
  , Data.Sequence.Seq
  , (Data.Sequence.<|)
  , (Data.Sequence.|>)
  , Data.Sequence.ViewL(EmptyL)
  , pattern ConsL
  , Data.Sequence.viewl
  , Data.Sequence.ViewR(EmptyR)
  , pattern SnocR
  , Data.Sequence.viewr
  , Data.Set.Set
    -- deepseq
  , module Control.DeepSeq
    -- extra
  , Control.Monad.Extra.whenJust
  , Control.Monad.Extra.whenJustM
  , Control.Monad.Extra.whileM
  , Control.Monad.Extra.partitionM
  , Control.Monad.Extra.concatMapM
  , Control.Monad.Extra.findM
  , Control.Monad.Extra.whenM
  , Control.Monad.Extra.unlessM
  , Control.Monad.Extra.orM
  , Control.Monad.Extra.andM
  , Control.Monad.Extra.anyM
  , Control.Monad.Extra.allM
    -- mtl
  , Control.Monad.Except.MonadError(..)
  , Control.Monad.Except.ExceptT(..)
  , Control.Monad.Except.Except
  , Control.Monad.Except.runExceptT
  , Control.Monad.Except.mapExceptT
  , Control.Monad.Except.withExceptT
  , Control.Monad.Except.runExcept
  , Control.Monad.Except.mapExcept
  , Control.Monad.Except.withExcept
  , Control.Monad.Reader.MonadReader
  , Control.Monad.Reader.Reader
  , Control.Monad.Reader.runReader
  , Control.Monad.Reader.ReaderT(..)
  , askReader
  , asksReader
  , localReader
  , module Control.Monad.ST
  , Control.Monad.State.MonadState
  , Control.Monad.State.State
  , Control.Monad.State.runState
  , Control.Monad.State.evalState
  , Control.Monad.State.execState
  , Control.Monad.State.withState
  , Control.Monad.State.StateT(..)
  , Control.Monad.State.execStateT
  , Control.Monad.State.evalStateT
  , getState
  , getsState
  , putState
  , modifyState
  , modifyState'
  , module Control.Monad.Trans
    -- stm
  , Control.Monad.STM.STM
  , atomicallySTM
  , retrySTM
  , checkSTM
  , Control.Monad.STM.throwSTM
  , Control.Monad.STM.catchSTM
  , module Control.Concurrent.STM.TVar
  , module Control.Concurrent.STM.TMVar
  , module Control.Concurrent.STM.TChan
  , module Control.Concurrent.STM.TQueue
  , module Control.Concurrent.STM.TBQueue
  , module Control.Concurrent.STM.TArray
    -- semigroups
  , Data.List.NonEmpty.NonEmpty
  , Data.Semigroup.Semigroup(..)
    -- text
  , Data.Text.Text
  , LText
  , Data.Text.Encoding.decodeUtf8
    -- transformers-base
  , module Control.Monad.Base

    -- Miscellaneous new functionality
  , eitherA
  , leftToMaybe
  , rightToMaybe
  , maybeToRight
  , maybeToLeft
  , maybeToBool
  , applyN
  , ordNub
  , notImplemented
  , ConvertibleStrings(..)
  ) where

import Control.Concurrent
import Control.Concurrent.Async
import Control.Concurrent.STM.TVar
import Control.Concurrent.STM.TMVar
import Control.Concurrent.STM.TChan
import Control.Concurrent.STM.TQueue
import Control.Concurrent.STM.TBQueue
import Control.Concurrent.STM.TArray
import Control.DeepSeq
import Control.Monad.Base
import Control.Monad.ST
import Control.Monad.Trans
import Data.Bifunctor
import Data.Bool
import Data.Char
import Data.Coerce
import Data.Either
import Data.Eq
import Data.Foldable hiding (foldl1, foldr1)
import Data.Functor.Identity
import Data.Int
import Data.IORef
import Data.Proxy
import Data.Traversable
import Data.Tuple
import GHC.Enum
import GHC.Float
import GHC.Num
import GHC.Real

import qualified Control.Applicative
import qualified Control.Exception
import qualified Control.Monad
import qualified Control.Monad.Except
import qualified Control.Monad.Extra
import qualified Control.Monad.Reader
import qualified Control.Monad.State
import qualified Control.Monad.STM
import qualified Data.ByteString
import qualified Data.ByteString.Char8
import qualified Data.ByteString.Builder
import qualified Data.ByteString.Lazy
import qualified Data.ByteString.Lazy.Char8
import qualified Data.Function
import qualified Data.Functor
import qualified Data.IntMap.Lazy
import qualified Data.IntMap.Strict
import qualified Data.IntSet
import qualified Data.List
import qualified Data.List.NonEmpty
import qualified Data.Map.Lazy
import qualified Data.Map.Strict
import qualified Data.Maybe
import qualified Data.Monoid
import qualified Data.Ord
import qualified Data.Semigroup
import qualified Data.Sequence
import qualified Data.Set
import qualified Data.Text
import qualified Data.Text.Encoding
import qualified Data.Text.IO
import qualified Data.Text.Lazy
import qualified Data.Text.Lazy.Builder
import qualified Data.Text.Lazy.Encoding
import qualified Data.Typeable
import qualified Debug.Trace
import qualified GHC.Base
import qualified GHC.Err
import qualified GHC.Exts
import qualified GHC.Generics
import qualified GHC.Num
import qualified GHC.Prim
import qualified GHC.Show
import qualified GHC.Types
import qualified System.IO
import qualified Text.Printf

--------------------------------------------------------------------------------
-- base

lift2 :: Control.Applicative.Applicative f => (a -> b -> c) -> f a -> f b -> f c
lift2 = Control.Applicative.liftA2

lift3 :: Control.Applicative.Applicative f => (a -> b -> c -> d) -> f a -> f b -> f c -> f d
lift3 = Control.Applicative.liftA3

lift4 :: Control.Applicative.Applicative f => (a -> b -> c -> d -> e) -> f a -> f b -> f c -> f d -> f e
lift4 f a b c d = f
  Control.Applicative.<$> a
  Control.Applicative.<*> b
  Control.Applicative.<*> c
  Control.Applicative.<*> d

lift5 :: Control.Applicative.Applicative f => (a -> b -> c -> d -> e -> e') -> f a -> f b -> f c -> f d -> f e -> f e'
lift5 f a b c d e = f
  Control.Applicative.<$> a
  Control.Applicative.<*> b
  Control.Applicative.<*> c
  Control.Applicative.<*> d
  Control.Applicative.<*> e

identity :: a -> a
identity = GHC.Base.id

map :: Data.Functor.Functor f => (a -> b) -> f a -> f b
map = Data.Functor.fmap

head :: Data.Foldable.Foldable f => f a -> Data.Maybe.Maybe a
head = Data.Foldable.foldr (\x _ -> Data.Maybe.Just x) Data.Maybe.Nothing

init :: [a] -> Data.Maybe.Maybe [a]
init [] = Data.Maybe.Nothing
init xs = Data.Maybe.Just (Data.List.init xs)

tail :: [a] -> Data.Maybe.Maybe [a]
tail []     = Data.Maybe.Nothing
tail (_:xs) = Data.Maybe.Just xs

last :: [a] -> Data.Maybe.Maybe a
last []     = Data.Maybe.Nothing
last [x]    = Data.Maybe.Just x
last (_:xs) = last xs

(!!) :: [a] -> GHC.Types.Int -> Data.Maybe.Maybe a
(!!) []     _    = Data.Maybe.Nothing
(!!) (x:[]) 0    = Data.Maybe.Just x
(!!) (_:xs) (!n) = xs !! (n GHC.Num.- 1)
infixl 9 !!

unsafeHead :: [a] -> a
unsafeHead = Data.List.head
{-# WARNING unsafeHead "'unsafeHead' remains in code" #-}

unsafeTail :: [a] -> [a]
unsafeTail = Data.List.tail
{-# WARNING unsafeTail "'unsafeTail' remains in code" #-}

unsafeInit :: [a] -> [a]
unsafeInit = Data.List.init
{-# WARNING unsafeInit "'unsafeInit' remains in code" #-}

unsafeLast :: [a] -> a
unsafeLast = Data.List.last
{-# WARNING unsafeLast "'unsafeLast' remains in code" #-}

unsafeIndex :: [a] -> GHC.Types.Int -> a
unsafeIndex = (Data.List.!!)
{-# WARNING unsafeIndex "'unsafeIndex' remains in code" #-}

unsafeFromJust :: Data.Maybe.Maybe a -> a
unsafeFromJust = Data.Maybe.fromJust
{-# WARNING unsafeFromJust "'unsafeFromJust' remains in code" #-}

zero :: Data.Monoid.Monoid a => a
zero = Data.Monoid.mempty

(++) :: Data.Monoid.Monoid a => a -> a -> a
(++) = Data.Monoid.mappend

undefined :: a
undefined = GHC.Err.undefined
{-# WARNING undefined "'undefined' remains in code" #-}

error :: GHC.Base.String -> a
error = GHC.Err.error
{-# WARNING error "'error' remains in code" #-}

-- | Renamed 'Debug.Trace.traceStack'.
trace :: GHC.Base.String -> a -> a
trace = Debug.Trace.traceStack
{-# WARNING trace "'trace' remains in code" #-}

-- | Renamed 'Debug.Trace.traceShowId'.
traceShow :: GHC.Show.Show a => a -> a
traceShow = Debug.Trace.traceShowId
{-# WARNING traceShow "'traceShow' remains in code" #-}

traceShowM :: (GHC.Show.Show a, Control.Monad.Monad m) => a -> m ()
traceShowM = Debug.Trace.traceShowM
{-# WARNING traceShowM "'traceShowM' remains in code" #-}

traceM :: Control.Monad.Monad m => GHC.Base.String -> m ()
traceM = Debug.Trace.traceM
{-# WARNING traceM "'traceM' remains in code" #-}

traceIO :: GHC.Base.String -> GHC.Types.IO ()
traceIO = Debug.Trace.traceIO
{-# WARNING traceIO "'traceIO' remains in code" #-}

show :: (GHC.Show.Show a, ConvertibleStrings GHC.Base.String b) => a -> b
show x = cs (GHC.Show.show x)

putStr :: MonadIO m => Data.Text.Text -> m ()
putStr = liftIO Data.Function.. Data.Text.IO.putStr

putStrLn :: MonadIO m => Data.Text.Text -> m ()
putStrLn = liftIO Data.Function.. Data.Text.IO.putStrLn

print :: (GHC.Show.Show a, MonadIO m) => a -> m ()
print = liftIO Data.Function.. System.IO.print

--------------------------------------------------------------------------------
-- bytestring

type LByteString = Data.ByteString.Lazy.ByteString

--------------------------------------------------------------------------------
-- containers

type LIntMap = Data.IntMap.Lazy.IntMap

type LMap = Data.Map.Lazy.Map

pattern ConsL x xs = x Data.Sequence.:< xs

pattern SnocR xs x = xs Data.Sequence.:> x

--------------------------------------------------------------------------------
-- mtl

askReader :: Control.Monad.Reader.MonadReader r m => m r
askReader = Control.Monad.Reader.ask

asksReader :: Control.Monad.Reader.MonadReader r m => (r -> a) -> m a
asksReader = Control.Monad.Reader.asks

localReader :: Control.Monad.Reader.MonadReader r m => (r -> r) -> m a -> m a
localReader = Control.Monad.Reader.local

getState :: Control.Monad.State.MonadState s m => m s
getState = Control.Monad.State.get

getsState :: Control.Monad.State.MonadState s m => (s -> t) -> m t
getsState = Control.Monad.State.gets

putState :: Control.Monad.State.MonadState s m => s -> m ()
putState = Control.Monad.State.put

modifyState :: Control.Monad.State.MonadState s m => (s -> s) -> m ()
modifyState = Control.Monad.State.modify

modifyState' :: Control.Monad.State.MonadState s m => (s -> s) -> m ()
modifyState' = Control.Monad.State.modify'

--------------------------------------------------------------------------------
-- stm

atomicallySTM :: Control.Monad.STM.STM a -> GHC.Types.IO a
atomicallySTM = Control.Monad.STM.atomically

retrySTM :: Control.Monad.STM.STM a
retrySTM = Control.Monad.STM.retry

checkSTM :: Bool -> Control.Monad.STM.STM ()
checkSTM = Control.Monad.STM.check

--------------------------------------------------------------------------------
-- text

type LText = Data.Text.Lazy.Text

--------------------------------------------------------------------------------
-- Miscellaneous new functionality

eitherA :: Control.Applicative.Alternative f => f a -> f b -> f (Data.Either.Either a b)
eitherA f g =
  Data.Functor.fmap Data.Either.Left f Control.Applicative.<|>
  Data.Functor.fmap Data.Either.Right g

leftToMaybe :: Data.Either.Either a b -> Data.Maybe.Maybe a
leftToMaybe = Data.Either.either Data.Maybe.Just (Data.Function.const Data.Maybe.Nothing)

rightToMaybe :: Data.Either.Either a b -> Data.Maybe.Maybe b
rightToMaybe = Data.Either.either (Data.Function.const Data.Maybe.Nothing) Data.Maybe.Just

maybeToRight :: a -> Data.Maybe.Maybe b -> Either a b
maybeToRight l = Data.Maybe.maybe (Left l) Right

maybeToLeft :: b -> Data.Maybe.Maybe a -> Data.Either.Either a b
maybeToLeft r = Data.Maybe.maybe (Data.Either.Right r) Data.Either.Left

maybeToBool :: Data.Maybe.Maybe a -> Data.Bool.Bool
maybeToBool Data.Maybe.Nothing = Data.Bool.False
maybeToBool _                  = Data.Bool.True

applyN :: Int -> (a -> a) -> a -> a
applyN 0 _ = Data.Function.id
applyN n f = f Data.Function.. applyN (n GHC.Num.- 1) f

-- | O(n * log n)
ordNub :: Data.Ord.Ord a => [a] -> [a]
ordNub l = go Data.Set.empty l
  where
    go _ [] = []
    go s (x:xs) =
      if x `Data.Set.member` s
      then go s xs
      else x : go (Data.Set.insert x s) xs

notImplemented :: a
notImplemented = GHC.Err.error "Not implemented"
{-# WARNING notImplemented "'notImplemented' remains in code" #-}

class ConvertibleStrings a b where
  cs :: a -> b

instance ConvertibleStrings GHC.Base.String GHC.Base.String where
  cs = GHC.Base.id

instance ConvertibleStrings GHC.Base.String Data.ByteString.ByteString where
  cs = Data.ByteString.Char8.pack

instance ConvertibleStrings GHC.Base.String Data.ByteString.Lazy.ByteString where
  cs = Data.ByteString.Lazy.Char8.pack

instance ConvertibleStrings GHC.Base.String Data.ByteString.Builder.Builder where
  cs = Data.ByteString.Builder.byteString Data.Function.. cs

instance ConvertibleStrings GHC.Base.String Data.Text.Text where
  cs = Data.Text.pack

instance ConvertibleStrings GHC.Base.String Data.Text.Lazy.Text where
  cs = Data.Text.Lazy.pack

instance ConvertibleStrings GHC.Base.String Data.Text.Lazy.Builder.Builder where
  cs = Data.Text.Lazy.Builder.fromString

instance ConvertibleStrings Data.ByteString.ByteString Data.ByteString.ByteString where
  cs = GHC.Base.id

instance ConvertibleStrings Data.ByteString.ByteString Data.ByteString.Lazy.ByteString where
  cs = Data.ByteString.Lazy.fromStrict

instance ConvertibleStrings Data.ByteString.ByteString Data.ByteString.Builder.Builder where
  cs = Data.ByteString.Builder.byteString

instance ConvertibleStrings Data.ByteString.Lazy.ByteString Data.ByteString.ByteString where
  cs = Data.ByteString.Lazy.toStrict

instance ConvertibleStrings Data.ByteString.Lazy.ByteString Data.ByteString.Lazy.ByteString where
  cs = GHC.Base.id

instance ConvertibleStrings Data.ByteString.Lazy.ByteString Data.ByteString.Builder.Builder where
  cs = Data.ByteString.Builder.lazyByteString

instance ConvertibleStrings Data.ByteString.Builder.Builder Data.ByteString.ByteString where
  cs = cs Data.Function.. Data.ByteString.Builder.toLazyByteString

instance ConvertibleStrings Data.ByteString.Builder.Builder Data.ByteString.Lazy.ByteString where
  cs = Data.ByteString.Builder.toLazyByteString

instance ConvertibleStrings Data.ByteString.Builder.Builder Data.ByteString.Builder.Builder where
  cs = GHC.Base.id

instance ConvertibleStrings Data.Text.Text GHC.Base.String where
  cs = Data.Text.unpack

instance ConvertibleStrings Data.Text.Text Data.ByteString.ByteString where
  cs = Data.Text.Encoding.encodeUtf8

instance ConvertibleStrings Data.Text.Text Data.ByteString.Lazy.ByteString where
  cs = cs Data.Function.. Data.Text.Encoding.encodeUtf8

instance ConvertibleStrings Data.Text.Text Data.ByteString.Builder.Builder where
  cs = cs Data.Function.. Data.Text.Encoding.encodeUtf8

instance ConvertibleStrings Data.Text.Text Data.Text.Text where
  cs = GHC.Base.id

instance ConvertibleStrings Data.Text.Text Data.Text.Lazy.Text where
  cs = Data.Text.Lazy.fromStrict

instance ConvertibleStrings Data.Text.Text Data.Text.Lazy.Builder.Builder where
  cs = Data.Text.Lazy.Builder.fromText

instance ConvertibleStrings Data.Text.Lazy.Text GHC.Base.String where
  cs = Data.Text.Lazy.unpack

instance ConvertibleStrings Data.Text.Lazy.Text Data.ByteString.ByteString where
  cs = cs Data.Function.. Data.Text.Lazy.Encoding.encodeUtf8

instance ConvertibleStrings Data.Text.Lazy.Text Data.ByteString.Lazy.ByteString where
  cs = Data.Text.Lazy.Encoding.encodeUtf8

instance ConvertibleStrings Data.Text.Lazy.Text Data.ByteString.Builder.Builder where
  cs = cs Data.Function.. Data.Text.Lazy.Encoding.encodeUtf8

instance ConvertibleStrings Data.Text.Lazy.Text Data.Text.Text where
  cs = Data.Text.Lazy.toStrict

instance ConvertibleStrings Data.Text.Lazy.Text Data.Text.Lazy.Text where
  cs = GHC.Base.id

instance ConvertibleStrings Data.Text.Lazy.Text Data.Text.Lazy.Builder.Builder where
  cs = Data.Text.Lazy.Builder.fromLazyText

instance ConvertibleStrings Data.Text.Lazy.Builder.Builder GHC.Base.String where
  cs = cs Data.Function.. Data.Text.Lazy.Builder.toLazyText

instance ConvertibleStrings Data.Text.Lazy.Builder.Builder Data.ByteString.ByteString where
  cs = cs Data.Function.. Data.Text.Lazy.Builder.toLazyText

instance ConvertibleStrings Data.Text.Lazy.Builder.Builder Data.ByteString.Lazy.ByteString where
  cs = cs Data.Function.. Data.Text.Lazy.Builder.toLazyText

instance ConvertibleStrings Data.Text.Lazy.Builder.Builder Data.ByteString.Builder.Builder where
  cs = cs Data.Function.. Data.Text.Lazy.Builder.toLazyText

instance ConvertibleStrings Data.Text.Lazy.Builder.Builder Data.Text.Text where
  cs = cs Data.Function.. Data.Text.Lazy.Builder.toLazyText

instance ConvertibleStrings Data.Text.Lazy.Builder.Builder Data.Text.Lazy.Text where
  cs = Data.Text.Lazy.Builder.toLazyText

instance ConvertibleStrings Data.Text.Lazy.Builder.Builder Data.Text.Lazy.Builder.Builder where
  cs = GHC.Base.id
