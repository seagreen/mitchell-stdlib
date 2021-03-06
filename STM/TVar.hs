{-# language CPP #-}

module STM.TVar
  ( -- * TVar
    TVar,
    newTVar,
    newTVarIO,
    readTVar,
    readTVarIO,
    writeTVar,
    modifyTVar,
    modifyTVar',
    swapTVar,
    mkWeakTVar,
  ) where

#ifdef DEP_UNLIFTIO
import UnliftIO.STM
#else
import Control.Concurrent.STM.TVar
#endif
