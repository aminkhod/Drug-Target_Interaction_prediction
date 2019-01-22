
#setwd("your\\dir")

#rm(list = ls())

## current data set name
dataset <- commandArgs(trailingOnly = TRUE)
nums = as.character(dataset)
w=read.table("KronRIsMKLW.txt")
db <- dataset
#db <- "nr"
setwd('../data/datasets')
switch (db,
        e = {
          cat("en data\n")
          flush.console()
          sd <- read.table("e_simmat_dc.txt")
          sd <- as.matrix(sd)
          st <- read.table("e_simmat_dg.txt")
          st <- as.matrix(st)
          Y <- read.table("e_admat_dgc.txt")
          Y <- as.matrix(Y) 
          Y <- t(Y)         
        },
        ic = {
          cat("ic data\n")
          flush.console()
          sd <- read.table("ic_simmat_dc.txt")
          sd <- as.matrix(sd)
          st <- read.table("ic_simmat_dg.txt")
          st <- as.matrix(st)
          Y <- read.table("ic_admat_dgc.txt")
          Y <- as.matrix(Y)
          Y <- t(Y)
        },
        gpcr = {
          cat("gpcr data\n")
          flush.console()
          sd <- read.table("gpcr_simmat_dc.txt")
          sd <- as.matrix(sd)
          st <- read.table("gpcr_simmat_dg.txt")
          st <- as.matrix(st)
          Y <- read.table("gpcr_admat_dgc.txt")
          Y <- as.matrix(Y)
          Y <- t(Y)
        },
        nr = {
          cat("nr data\n")
          flush.console()
          sd <- read.table("nr_simmat_dc.txt")
          sd <- as.matrix(sd)
          st <- read.table("nr_simmat_dg.txt")
          st <- as.matrix(st)
          Y <- read.table("nr_admat_dgc.txt")
          Y <- as.matrix(Y)
          Y <- t(Y)
        },
        srep = {
          cat("Scientific Reports data\n")
          flush.console()
          sd <- read.table("drug ChemSimilarity.txt", sep = ",")
          sd <- data.matrix(sd)
          st <- read.table("target SeqSimilarity.txt", sep = ",")
          st <- data.matrix(st)
          Y <- read.table("adjacent Matrix.txt", sep = ",")
          Y <- data.matrix(Y)
        },
        stop("db should be one of the follows: 
             {e, ic, gpcr, nr or srep}\n")
        )
setwd('../')
setwd('../')
setwd('PyDTI/')

#It use new dataset that its name is "KD"
if (db == "kd") {
  dim(Y) ## 68 * 442
  dim(sd) ## 68 * 68
  dim(st) ##  442 * 442
  
  
  idxZeroCol <- which(colSums(Y) == 0)
  Y <- Y[, -idxZeroCol]
  st <- st[-idxZeroCol, -idxZeroCol]
  
  ## which(colSums(Y) == 0)
  
  idxZeroRow <- which(rowSums(Y) == 0)
  Y <- Y[-idxZeroRow, ]
  sd <- sd[-idxZeroRow, -idxZeroRow]
  
  which(rowSums(Y) == 0)
  which(colSums(Y) == 0)
  
  #dim(Y)  ## 65 373
  #dim(sd) ## 65 65
  #dim(st) ## 373 373
  
  #sd[1:3, 1:3]
  #st[1:3, 1:3]
  #Y[1:3, 1:3]
}




## load required packages
#install.packages("matrixcalc")
#install.packages("data.table")
#install.packages("Rcpp")
#install.packages("ROCR")
#install.packages("Bolstad2")
#install.packages("MESS")
#install.packages("nloptr")

pkgs <- c("matrixcalc", "data.table", "Rcpp", "ROCR", 
          "Bolstad2", "MESS", "nloptr")
rPkgs <- lapply(pkgs, require, character.only = TRUE)

## source required R files
rSourceNames <- c(
#  "doCVPositiveOnly.R",
#  "doCVPositiveOnly3.R",
#  "doCrossVal.R",
#  "calAUPR.R",
#  "evalMetrics.R",
  "combineKernels.R",
  "eigDecomp.R",
  "kronRls.R" ,
  "kronRlsC.R",
  "kronRlsMKL.R" ,
  "optWeights.R"
)
rSN <- lapply(rSourceNames, source, verbose = FALSE)

## sourceCPP required C++ files
cppSourceNames <- c("fastKF.cpp", "fastKgipMat.cpp", 
                    "log1pexp.cpp", "sigmoid.cpp")
cppSN <- lapply(cppSourceNames, sourceCpp, verbose = FALSE)


## convert to kernel
isKernel <- TRUE
if (isKernel) {
  if (!isSymmetric(sd)) {
    sd <- (sd + t(sd)) / 2
  }
  epsilon <- 0.1
  while (!is.positive.semi.definite(sd)) {
    sd <- sd + epsilon * diag(nrow(sd))
  }
  if (!isSymmetric(st)) {
    st <- (st + t(st)) / 2
  }
  epsilon <- 0.1
  while (!is.positive.semi.definite(st)) {
    st <- st + epsilon * diag(nrow(st))
  }
}

Y <- t(Y)
tmp <- sd
sd <- st
st <- tmp

#Y[1:3, 1:3]


## do cross-validation
#kfold <- 10
#numSplit <- 5

## DT-Hybrid method
#savedFolds <- doCrossVal(Y, nfold = kfold, nsplit = numSplit)
#savedFolds <- doCrossVal(Y, nfold = 10, nsplit = 5)
## for saving results
#AUPRVec <- vector(length = kfold)
#AUCVec <- vector(length = kfold)
#finalResult <- matrix(NA, nrow = numSplit, ncol = 2)
#colnames(finalResult) <- c("AUPR", "AUC")


## alpha and beta
#resAB <- matrix(NA, nrow = kfold, ncol = 4)
resAB <- matrix(NA, nrow = 1, ncol = 4)
colnames(resAB) <- c("optAlpha1", "optAlpha2", "optBeta1", "optBeta2")
resAB <- as.data.frame(resAB)
#finalAB <- vector("list", length = numSplit)
finalAB <- vector("list", length = 1)
## main loop
#for (i in 1:numSplit) {
#  for (j in 1:kfold) {
#    cat("numSplit:", i, "/", numSplit, ";", "kfold:", j, 
#        "/", kfold, "\n")
#    flush.console()

### training set with the test set links removed

#    Yfold <- savedFolds[[i]][[j]][[1]]
w=t(w)
Yfold <-w*Y
KgipD <- fastKgipMat(Yfold, 1)
KgipT <- fastKgipMat(t(Yfold), 1)

## extract test set
#testSet <- savedFolds[[i]][[j]][[2]]
  

lmd <- 1
sgm <- 0.25
maxiter <- 20

## kronrlsMKL
MKL <- kronRlsMKL(
  K1 = list(sd = sd, KgipD = KgipD),
  K2 = list(st = st, KgipT = KgipT),
  Yfold = Yfold,
  lmd = lmd,
  sgm = sgm,
  maxiter = maxiter
)

Ypred <- MKL$Yhat
#resAB[j, 1:2] <- MKL$alph
resAB[1, 1:2] <- MKL$alph
#resAB[j, 3:4] <- MKL$bta
resAB[1, 3:4] <- MKL$bta

#testLabel <- Y[testSet]
#score <- Ypred[testSet]
score <- t(Ypred)
write.table(score,"KronRIsMKLscore.txt", col.names = , row.names = F)
#result <- calAUPR(testLabel, score)

#AUPRVec[j] <- result[1, "aupr"]
#AUCVec[j] <- result[1, "auc"]
#}
#AUPR <- mean(AUPRVec)
#AUC <- mean(AUCVec)
#finalResult[i, "AUPR"] <- AUPR
#finalResult[i, "AUC"] <- AUC

#finalAB[[i]] <- resAB
finalAB[[1]] <- resAB
#}
cat(
  "\n======================\n\n",
  "db is: ", db, "\n",
  ## hyper-parameters
  "Alpha 1 = ", resAB$optAlpha1, "\n",
  "Alpha 2 = ", resAB$optAlpha2, "\n",
  "Beta 1 = ", resAB$optBeta1, "\n",
  "Beta 2 = ", resAB$optBeta2, "\n",
  "\n=====================\n")


#auc <- round(mean(finalResult[, "AUC"]), 3)
#aucSD <- round(sd(finalResult[, "AUC"]), 3)

#aupr <- round(mean(finalResult[, "AUPR"]), 3)
#auprSD <- round(sd(finalResult[, "AUPR"]), 3)



# save to file
#curDate <- format(Sys.time(), format = "%Y-%m-%d")
#curTime <- format(Sys.time(), format =  "%H.%M.%S")
#savedFileName <- paste0(db, "_", curDate, "_", curTime, "_auc", auc, "+-", aucSD, "_aupr", aupr, "+-", auprSD, ".RData")
#cat("\n\n")
#print(savedFileName)
#save.image(file = savedFileName)

