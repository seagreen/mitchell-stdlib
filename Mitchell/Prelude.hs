{-# LANGUAGE AllowAmbiguousTypes                #-}
{-# LANGUAGE DefaultSignatures                  #-}
{-# LANGUAGE BangPatterns                       #-}
{-# LANGUAGE FlexibleContexts                   #-}
{-# LANGUAGE FlexibleInstances                  #-}
{-# LANGUAGE FunctionalDependencies             #-}
{-# LANGUAGE IncoherentInstances                #-}
{-# LANGUAGE MultiParamTypeClasses              #-}
{-# LANGUAGE PatternSynonyms                    #-}
{-# LANGUAGE RankNTypes                         #-}
{-# LANGUAGE RoleAnnotations                    #-}
{-# LANGUAGE ScopedTypeVariables                #-}
{-# LANGUAGE TypeApplications                   #-}
{-# LANGUAGE TypeFamilies                       #-}
{-# LANGUAGE TypeSynonymInstances               #-}
{-# LANGUAGE ViewPatterns                       #-}
{-# OPTIONS_GHC -fno-warn-missing-import-lists  #-}
{-# OPTIONS_GHC -fno-warn-redundant-constraints #-}
{-# OPTIONS_GHC -fno-warn-unsafe                #-}

module Mitchell.Prelude
  ( -- * async
    module Control.Concurrent.Async
    -- * base
    -- ** Control.Applicative
  , Applicative(..)
  , Control.Applicative.Const(..)
  , lift2
  , lift3
  , lift4
  , lift5
  , Control.Applicative.Alternative(..)
  , Control.Applicative.optional
  , eitherA
    -- ** Control.Category
  , Control.Category.Category
  , (Control.Category..)
  , identity
    -- ** Control.Concurrent
  , Control.Concurrent.ThreadId
  , Control.Concurrent.myThreadId
  , Control.Concurrent.threadDelay
    -- ** Control.Exception
  , Control.Exception.SomeException(..)
  , Control.Exception.Exception(..)
  , Control.Exception.evaluate
  , Control.Exception.assert
  , assertM
    -- ** Control.Monad
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
    -- ** Data.Bifunctor
  , module Data.Bifunctor
    -- ** Data.Bool
  , module Data.Bool
    -- ** Data.Char
  , module Data.Char
    -- ** Data.Coerce
  , module Data.Coerce
    -- ** Data.Either
  , module Data.Either
  , leftToMaybe
  , rightToMaybe
    -- ** Data.Eq
  , module Data.Eq
    -- ** Data.Foldable
  , module Data.Foldable
    -- ** Data.Function
  , Data.Function.const
  , Data.Function.flip
  , (Data.Function.$)
  , (Data.Function.&)
  , Data.Function.fix
  , Data.Function.on
  , applyN
    -- ** Data.Functor
  , Data.Functor.Functor((<$))
  , (Data.Functor.$>)
  , (Data.Functor.<$>)
  , Data.Functor.void
  , Mitchell.Prelude.map
  , module Data.Functor.Identity
    -- ** Data.Int
  , module Data.Int
    -- ** Data.IORef
  , module Data.IORef
    -- ** Data.List
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
  , (Mitchell.Prelude.!!)
  , Mitchell.Prelude.head
  , Mitchell.Prelude.init
  , Mitchell.Prelude.last
  , Mitchell.Prelude.tail
  , ordNub
  , unsafeHead
  , unsafeTail
  , unsafeInit
  , unsafeLast
  , unsafeIndex
    -- ** Data.List.NonEmpty
  , Data.List.NonEmpty.NonEmpty
    -- ** Data.Maybe
  , Maybe(..)
  , maybe
  , Data.Maybe.isJust
  , Data.Maybe.isNothing
  , Data.Maybe.fromMaybe
  , Data.Maybe.listToMaybe
  , Data.Maybe.maybeToList
  , Data.Maybe.catMaybes
  , Data.Maybe.mapMaybe
  , unsafeFromJust
  , maybeToRight
  , maybeToLeft
    -- ** Data.Monoid
  , Monoid
  , zero
    -- ** Data.Ord
  , Ord(..)
  , Ordering(..)
  , Data.Ord.comparing
    -- ** Data.Proxy
  , module Data.Proxy
    -- ** Data.Semigroup
  , Data.Semigroup.Semigroup
  , (Mitchell.Prelude.++)
    -- ** Data.Traversable
  , module Data.Traversable
    -- ** Data.Tuple
  , module Data.Tuple
    -- ** Data.Typeable
  , Data.Typeable.Typeable
  , Data.Typeable.TypeRep
  , Data.Typeable.typeRep
    -- ** Debug.Trace
  , trace
  , traceM
  , traceIO
  , traceShow
  , traceShowM
    -- ** GHC.Base
  , ($!)
    -- ** GHC.Enum
  , module GHC.Enum
    -- ** GHC.Err
  , Mitchell.Prelude.undefined
  , Mitchell.Prelude.error
  , notImplemented
    -- ** GHC.Exts
  , GHC.Exts.Constraint
  , GHC.Exts.IsList(fromList)
    -- ** GHC.Float
  , module GHC.Float
    -- ** GHC.Generics
  , GHC.Generics.Generic
    -- ** GHC.Num
  , module GHC.Num
    -- ** GHC.Prim
  , GHC.Prim.seq
    -- ** GHC.Real
  , module GHC.Real
    -- ** GHC.Show
  , Show
  , Mitchell.Prelude.show
    -- ** GHC.IO
  , GHC.IO.IO
    -- ** System.IO
  , Mitchell.Prelude.putStr
  , Mitchell.Prelude.putStrLn
  , Mitchell.Prelude.print
    -- ** Text.Printf
  , Text.Printf.printf
  , Text.Printf.hPrintf
    -- * bytestring
  , Data.ByteString.ByteString
  , LByteString
    -- * containers
    -- ** Data.IntMap.Strict
  , Data.IntMap.Strict.IntMap
    -- ** Data.IntMap.Lazy
  , LIntMap
    -- ** Data.IntSet
  , Data.IntSet.IntSet
    -- ** Data.Map.Strict
  , Data.Map.Strict.Map
    -- ** Data.Map.Lazy
  , LMap
    -- ** Data.Sequence
  , Data.Sequence.Seq
  , pattern (:>)
  , pattern (:<)
    -- ** Data.Set
  , Data.Set.Set
    -- * deepseq
  , module Control.DeepSeq
    -- * extra
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
    -- * microlens
  , Lens.Micro.Platform.Lens
  , Lens.Micro.Platform.Lens'
  , Lens.Micro.Platform.Traversal
  , Lens.Micro.Platform.Traversal'
  , Lens.Micro.Platform.lens
  , Lens.Micro.Platform.over
  , Lens.Micro.Platform.set
  , Lens.Micro.Platform.mapped
  , Lens.Micro.Platform.to
  , Lens.Micro.Platform.view
  , Lens.Micro.Platform.preview
  , viewAll
  , Lens.Micro.Platform.folded
  , Lens.Micro.Platform.at
  , Lens.Micro.Platform._1
  , Lens.Micro.Platform._2
  , Lens.Micro.Platform._3
  , Lens.Micro.Platform._4
  , Lens.Micro.Platform._5
  , Lens.Micro.Platform.traversed
  , Lens.Micro.Platform.each
  , Lens.Micro.Platform.ix
  , Lens.Micro.Platform._head
  , Lens.Micro.Platform._tail
  , Lens.Micro.Platform._init
  , Lens.Micro.Platform._last
  , Lens.Micro.Platform._Left
  , Lens.Micro.Platform._Right
  , Lens.Micro.Platform._Just
  , Lens.Micro.Platform._Nothing
    -- * mtl
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
    -- * stm
  , Control.Monad.STM.STM
  , atomicallySTM
  , retrySTM
  , checkSTM
  , Control.Monad.STM.throwSTM
  , Control.Monad.STM.catchSTM
  , module Control.Concurrent.STM.TVar
  , readTVarCheck
  , module Control.Concurrent.STM.TMVar
  , module Control.Concurrent.STM.TChan
  , module Control.Concurrent.STM.TQueue
  , module Control.Concurrent.STM.TBQueue
  , module Control.Concurrent.STM.TArray
    -- * safe-exceptions
  , Control.Exception.Safe.throw
  , Control.Exception.Safe.throwTo
  , Control.Exception.Safe.impureThrow
  , Control.Exception.Safe.catch
  , Control.Exception.Safe.catchAny
  , Control.Exception.Safe.catchAsync
  , Control.Exception.Safe.handle
  , Control.Exception.Safe.handleAny
  , Control.Exception.Safe.handleAsync
  , Control.Exception.Safe.try
  , Control.Exception.Safe.tryAny
  , Control.Exception.Safe.tryAsync
  , Control.Exception.Safe.onException
  , Control.Exception.Safe.bracket
  , Control.Exception.Safe.bracket_
  , Control.Exception.Safe.finally
  , Control.Exception.Safe.withException
  , Control.Exception.Safe.bracketOnError
  , Control.Exception.Safe.bracketOnError_
  , Control.Exception.Safe.MonadThrow
  , Control.Exception.Safe.MonadCatch
  , Control.Exception.Safe.MonadMask(..)
  , Control.Exception.Safe.mask_
  , Control.Exception.Safe.uninterruptibleMask_
  , Control.Exception.Safe.catchIOError
  , Control.Exception.Safe.handleIOError
  , Control.Exception.Safe.IOException
    -- * text
  , Data.Text.Text
  , LText
    -- * transformers-base
  , module Control.Monad.Base
  , module Control.Monad.Trans.Class
  , Control.Monad.IO.Class.MonadIO
  , io
    -- * unordered-containers
  , Data.HashMap.Strict.HashMap
  , LHashMap
  , Data.HashSet.HashSet
    -- * Throws
  , Throws
  , throwChecked
  , catchChecked
    -- * Misc
  , pack, unpack, unsafeUnpack
  , lazy, strict
  , encodeUtf8, decodeUtf8, unsafeDecodeUtf8
  ) where

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
import Control.Monad.Trans.Class
import Data.Bifunctor
import Data.Coerce
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
import Prelude

import qualified Control.Applicative
import qualified Control.Category
import qualified Control.Concurrent
import qualified Control.Exception
import qualified Control.Exception.Safe
import qualified Control.Monad
import qualified Control.Monad.Except
import qualified Control.Monad.Extra
import qualified Control.Monad.IO.Class
import qualified Control.Monad.Reader
import qualified Control.Monad.State
import qualified Control.Monad.STM
import qualified Data.Bool
import qualified Data.ByteString
import qualified Data.ByteString.Lazy
import qualified Data.Char
import qualified Data.Either
import qualified Data.Eq
import qualified Data.Foldable
import qualified Data.Function
import qualified Data.Functor
import qualified Data.HashMap.Lazy
import qualified Data.HashMap.Strict
import qualified Data.HashSet
import qualified Data.IntMap.Lazy
import qualified Data.IntMap.Strict
import qualified Data.IntSet
import qualified Data.List
import qualified Data.List.NonEmpty
import qualified Data.Map.Lazy
import qualified Data.Map.Strict
import qualified Data.Maybe
import qualified Data.Ord
import qualified Data.Semigroup
import qualified Data.Sequence
import qualified Data.Set
import qualified Data.Text
import qualified Data.Text.Encoding
import qualified Data.Text.Encoding.Error
import qualified Data.Text.IO
import qualified Data.Text.Lazy
import qualified Data.Text.Lazy.Encoding
import qualified Data.Typeable
import qualified Debug.Trace
import qualified GHC.Exts
import qualified GHC.Generics
import qualified GHC.IO
import qualified GHC.Prim
import qualified Lens.Micro.Platform
import qualified Text.Printf

--------------------------------------------------------------------------------
-- base

lift2 :: Applicative f => (a -> b -> c) -> f a -> f b -> f c
lift2 = Control.Applicative.liftA2
{-# INLINE lift2 #-}

lift3 :: Applicative f => (a -> b -> c -> d) -> f a -> f b -> f c -> f d
lift3 = Control.Applicative.liftA3
{-# INLINE lift3 #-}

lift4 :: Applicative f => (a -> b -> c -> d -> e) -> f a -> f b -> f c -> f d -> f e
lift4 f a b c d = f <$> a <*> b <*> c <*> d
{-# INLINE lift4 #-}

lift5 :: Applicative f => (a -> b -> c -> d -> e -> e') -> f a -> f b -> f c -> f d -> f e -> f e'
lift5 f a b c d e = f <$> a <*> b <*> c <*> d <*> e
{-# INLINE lift5 #-}

eitherA :: Control.Applicative.Alternative f => f a -> f b -> f (Either a b)
eitherA f g = fmap Left f Control.Applicative.<|> fmap Right g
{-# INLINE eitherA #-}

leftToMaybe :: Either a b -> Maybe a
leftToMaybe = either Just (const Nothing)
{-# INLINE leftToMaybe #-}

rightToMaybe :: Either a b -> Maybe b
rightToMaybe = either (const Nothing) Just
{-# INLINE rightToMaybe #-}

maybeToRight :: a -> Maybe b -> Either a b
maybeToRight l = maybe (Left l) Right
{-# INLINE maybeToRight #-}

maybeToLeft :: b -> Maybe a -> Either a b
maybeToLeft r = maybe (Right r) Left
{-# INLINE maybeToLeft #-}

identity :: Control.Category.Category p => p a a
identity = Control.Category.id
{-# INLINE identity #-}

assertM :: Control.Monad.Monad m => Bool -> m ()
assertM b = Control.Exception.assert b (pure ())
{-# INLINE assertM #-}

applyN :: Int -> (a -> a) -> a -> a
applyN 0 _ = id
applyN n f = f . applyN (n - 1) f
{-# INLINE applyN #-}

map :: Functor f => (a -> b) -> f a -> f b
map = fmap
{-# INLINE map #-}

head :: Foldable f => f a -> Maybe a
head = foldr (\x _ -> Just x) Nothing
{-# INLINE head #-}

init :: [a] -> Maybe [a]
init [] = Nothing
init xs = Just (Prelude.init xs)
{-# INLINE init #-}

tail :: [a] -> Maybe [a]
tail []     = Nothing
tail (_:xs) = Just xs
{-# INLINE tail #-}

last :: [a] -> Maybe a
last []     = Nothing
last [x]    = Just x
last (_:xs) = Mitchell.Prelude.last xs
{-# INLINE last #-}

(!!) :: [a] -> Int -> Maybe a
(!!) []     _    = Nothing
(!!) (x:[]) 0    = Just x
(!!) (_:xs) (!n) = xs Mitchell.Prelude.!! (n - 1)
infixl 9 !!
{-# INLINE (!!) #-}

-- | O(n * log n)
ordNub :: forall a. Ord a => [a] -> [a]
ordNub l = go Data.Set.empty l
 where
  go :: Data.Set.Set a -> [a] -> [a]
  go _ [] = []
  go s (x:xs) =
    if x `Data.Set.member` s
    then go s xs
    else x : go (Data.Set.insert x s) xs

unsafeHead :: [a] -> a
unsafeHead = Prelude.head
{-# WARNING unsafeHead "'unsafeHead' remains in code" #-}

unsafeTail :: [a] -> [a]
unsafeTail = Prelude.tail
{-# WARNING unsafeTail "'unsafeTail' remains in code" #-}

unsafeInit :: [a] -> [a]
unsafeInit = Prelude.init
{-# WARNING unsafeInit "'unsafeInit' remains in code" #-}

unsafeLast :: [a] -> a
unsafeLast = Prelude.last
{-# WARNING unsafeLast "'unsafeLast' remains in code" #-}

unsafeIndex :: [a] -> Int -> a
unsafeIndex = (Prelude.!!)
{-# WARNING unsafeIndex "'unsafeIndex' remains in code" #-}

unsafeFromJust :: Maybe a -> a
unsafeFromJust = Data.Maybe.fromJust
{-# WARNING unsafeFromJust "'unsafeFromJust' remains in code" #-}

zero :: Monoid a => a
zero = mempty
{-# INLINE zero #-}

(++) :: Data.Semigroup.Semigroup a => a -> a -> a
(++) = (Data.Semigroup.<>)
{-# INLINE (++) #-}

-- | 'error' with a warning.
undefined :: a
undefined = Prelude.undefined
{-# WARNING undefined "'undefined' remains in code" #-}

-- | 'error' with a warning.
error :: String -> a
error = Prelude.error
{-# WARNING error "'error' remains in code" #-}

notImplemented :: a
notImplemented = Prelude.error "Not implemented"
{-# WARNING notImplemented "'notImplemented' remains in code" #-}


-- | Renamed 'Debug.Trace.traceStack'.
trace :: String -> a -> a
trace = Debug.Trace.traceStack
{-# WARNING trace "'trace' remains in code" #-}

-- | Renamed 'Debug.Trace.traceShowId'.
traceShow :: Show a => a -> a
traceShow = Debug.Trace.traceShowId
{-# WARNING traceShow "'traceShow' remains in code" #-}

traceShowM :: (Show a, Control.Monad.Monad m) => a -> m ()
traceShowM = Debug.Trace.traceShowM
{-# WARNING traceShowM "'traceShowM' remains in code" #-}

traceM :: Control.Monad.Monad m => String -> m ()
traceM = Debug.Trace.traceM
{-# WARNING traceM "'traceM' remains in code" #-}

traceIO :: String -> IO ()
traceIO = Debug.Trace.traceIO
{-# WARNING traceIO "'traceIO' remains in code" #-}

show :: (Show a, PackOrId b) => a -> b
show x = packOrId (Prelude.show x)
{-# INLINE show #-}

putStr :: Control.Monad.IO.Class.MonadIO m => Data.Text.Text -> m ()
putStr = io . Data.Text.IO.putStr
{-# INLINE putStr #-}

putStrLn :: Control.Monad.IO.Class.MonadIO m => Data.Text.Text -> m ()
putStrLn = io . Data.Text.IO.putStrLn
{-# INLINE putStrLn #-}

print :: (Show a, Control.Monad.IO.Class.MonadIO m) => a -> m ()
print = io . Prelude.print
{-# INLINE print #-}

--------------------------------------------------------------------------------
-- bytestring

type LByteString = Data.ByteString.Lazy.ByteString

--------------------------------------------------------------------------------
-- containers

type LIntMap = Data.IntMap.Lazy.IntMap

type LMap = Data.Map.Lazy.Map

pattern (:<) :: a -> Data.Sequence.Seq a -> Data.Sequence.Seq a
pattern (:<) x xs <- (Data.Sequence.viewl -> x Data.Sequence.:< xs) where
  (:<) x xs = x Data.Sequence.<| xs

pattern (:>) :: Data.Sequence.Seq a -> a -> Data.Sequence.Seq a
pattern (:>) xs x <- (Data.Sequence.viewr -> xs Data.Sequence.:> x) where
  (:>) xs x = xs Data.Sequence.|> x

--------------------------------------------------------------------------------
-- microlens

-- | A re-renamed (^..)/toListOf (I hate toListOf)
viewAll :: Lens.Micro.Platform.Getting (Data.Semigroup.Endo [a]) s a -> s -> [a]
viewAll = Lens.Micro.Platform.toListOf

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

atomicallySTM :: Control.Monad.IO.Class.MonadIO m => Control.Monad.STM.STM a -> m a
atomicallySTM = io . Control.Monad.STM.atomically
{-# INLINE atomicallySTM #-}

retrySTM :: Control.Monad.STM.STM a
retrySTM = Control.Monad.STM.retry
{-# INLINE retrySTM #-}

checkSTM :: Bool -> Control.Monad.STM.STM ()
checkSTM = Control.Monad.STM.check
{-# INLINE checkSTM #-}

readTVarCheck :: (a -> Bool) -> Control.Concurrent.STM.TVar.TVar a -> Control.Monad.STM.STM a
readTVarCheck f var = do
  x <- Control.Concurrent.STM.TVar.readTVar var
  checkSTM (f x)
  pure x

--------------------------------------------------------------------------------
-- text

type LText = Data.Text.Lazy.Text

--------------------------------------------------------------------------------
-- transformers

io :: Control.Monad.IO.Class.MonadIO m => IO a -> m a
io = Control.Monad.IO.Class.liftIO
{-# INLINE io #-}

--------------------------------------------------------------------------------
-- unordered-containers

type LHashMap = Data.HashMap.Lazy.HashMap

--------------------------------------------------------------------------------
-- Awesome 'Throws' machinery from
-- http://www.well-typed.com/blog/2015/07/checked-exceptions/

type role Throws representational
class Throws e

instance Throws (Catch e)

newtype Wrap e a = Wrap { unWrap :: Throws e => a }

newtype Catch e = Catch e

coerceWrap :: Wrap e a -> Wrap (Catch e) a
coerceWrap = Data.Coerce.coerce

unthrow :: forall e a. (Throws e => a) -> a
unthrow = unWrap . coerceWrap @e . Wrap


throwChecked
  :: ( Control.Exception.Safe.MonadThrow m
     , Control.Exception.Safe.Exception e
     , Throws e
     )
  => e -> m a
throwChecked = Control.Exception.Safe.throw

catchChecked
  :: forall m e a.
     ( Control.Exception.Safe.MonadCatch m
     , Control.Exception.Safe.Exception e
     )
  => (Throws e => m a)
  -> (e -> m a)
  -> m a
catchChecked act = Control.Exception.Safe.catch (unthrow @e act)

--------------------------------------------------------------------------------
-- Misc

class PackUnpack t where
  type UnpackConstraint t :: GHC.Exts.Constraint
  pack :: String -> t
  unpack :: UnpackConstraint t => t -> String

class UnsafeUnpack t where
  unsafeUnpack :: t -> String
{-# WARNING unsafeUnpack "'unsafeUnpack' remains in code" #-}

instance PackUnpack Data.ByteString.ByteString where
  type UnpackConstraint Data.ByteString.ByteString = Throws Data.Text.Encoding.Error.UnicodeException
  pack = Data.Text.Encoding.encodeUtf8 . pack
  unpack = Data.Text.unpack . Data.Text.Encoding.decodeUtf8

instance UnsafeUnpack Data.ByteString.ByteString where
  unsafeUnpack = Data.Text.unpack . Data.Text.Encoding.decodeUtf8

instance PackUnpack LByteString where
  type UnpackConstraint LByteString = Throws Data.Text.Encoding.Error.UnicodeException
  pack = Data.Text.Lazy.Encoding.encodeUtf8 . pack
  unpack = Data.Text.Lazy.unpack . Data.Text.Lazy.Encoding.decodeUtf8

instance UnsafeUnpack LByteString where
  unsafeUnpack = Data.Text.Lazy.unpack . Data.Text.Lazy.Encoding.decodeUtf8

instance PackUnpack Data.Text.Text where
  type UnpackConstraint Data.Text.Text = ()
  pack = Data.Text.pack
  unpack = Data.Text.unpack

instance PackUnpack LText where
  type UnpackConstraint LText = ()
  pack = Data.Text.Lazy.pack
  unpack = Data.Text.Lazy.unpack


-- Throwaway class that only exists to add a String instance to PackUnpack. This
-- is because although we don't want packing/unpacking String to typecheck
-- (because it's just identity), we do want our generalized "show" to be able to
-- target Strings, if desired.
class PackOrId t where
  packOrId :: String -> t
  default packOrId :: PackUnpack t => String -> t
  packOrId = pack

instance PackOrId String where
  packOrId = identity

instance PackOrId Data.ByteString.ByteString
instance PackOrId LByteString
instance PackOrId Data.Text.Text
instance PackOrId LText


-- | Generalize encode/decode UTF-8.
class EncodeDecodeUtf8 t bs | t -> bs, bs -> t where
  encodeUtf8 :: t -> bs
  decodeUtf8 :: Throws Data.Text.Encoding.Error.UnicodeException => bs -> t
  unsafeDecodeUtf8 :: bs -> t
{-# WARNING unsafeDecodeUtf8 "'unsafeDecodeUtf8' remains in code" #-}

instance EncodeDecodeUtf8 Data.Text.Text Data.ByteString.ByteString where
  encodeUtf8 = Data.Text.Encoding.encodeUtf8
  decodeUtf8 = Data.Text.Encoding.decodeUtf8
  unsafeDecodeUtf8 = Data.Text.Encoding.decodeUtf8

instance EncodeDecodeUtf8 LText LByteString where
  encodeUtf8 = Data.Text.Lazy.Encoding.encodeUtf8
  decodeUtf8 = Data.Text.Lazy.Encoding.decodeUtf8
  unsafeDecodeUtf8 = Data.Text.Lazy.Encoding.decodeUtf8


-- | Generalize lazify/strictify.
class LazyStrict l s | l -> s, s -> l where
  lazy :: s -> l
  strict :: l -> s

instance LazyStrict LByteString Data.ByteString.ByteString where
  lazy = Data.ByteString.Lazy.fromStrict
  strict = Data.ByteString.Lazy.toStrict

instance LazyStrict LText Data.Text.Text where
  lazy = Data.Text.Lazy.fromStrict
  strict = Data.Text.Lazy.toStrict
