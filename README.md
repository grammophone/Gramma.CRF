# Gramma.CRF
This .NET library offers Linear-Chain Conditional Random Fields implementations. Two flavors are provided: Abstract classes `LinearChainCRF<I, T>` and `ConstrainedLinearChainCRF<I, T>`. The `I` is the type of input, all of it, not an element of an input sequence. This is because the library's implementation can access at any time every part of `I` instances which need not even be sequences; they can be any kind of object. On the other hand, `T` is the type of the element of the output sequence. The `LinearChainCRF<I, T>` is a classic implementation; the derived `ConstrainedLinearChainCRF<I, T>` takes into account the sparsity of the allowed bigrams in the output sequence of `T` elements in order to speed up computations. Details about the latter will be available in a paper shortly.

In order to use the library, create a concrete descendant of either abstract class specifying the arguments needed by its base constructor. Then train the Conditional Random Field using either `OnlineTrain` or `OfflineTrain` methods. You can save and load the trained Conditional Random Field using standard .NET serialization of the concrete descendant, which must also be serializable. Sequence predictions can then be performed using method `GetSequenceEvaluator` which takes the input of type `I` as its argument. The returned `LinearChainCRF<I, T>.SequenceEvaluator` offers properties and methods which provide the predicted sequence and statistical figures.

In general, these implementations follow the notes of Charles Elkan's
["Log-Linear Models and Conditional Random Fields"](http://www.cs.columbia.edu/~smaskey/CS6998-0412/supportmaterial/cikmtutorial.pdf)
with own corrections included, combined with forward and backward vectors scaling for arithmetic 
robustness as described in Rabiner's 
["A tutorial on hidden markov models and selected apllications in speech recognition"]
(http://www.cs.cornell.edu/Courses/cs4758/2012sp/materials/hmm_paper_rabiner.pdf) with 
corrections provided by Rahimi's ["An Erratum 
for 'A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition'"](http://alumni.media.mit.edu/~rahimi/rabiner/rabiner-errata/rabiner-errata.html).
