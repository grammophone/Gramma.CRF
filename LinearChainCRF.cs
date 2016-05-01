using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Grammophone.Optimization.QuasiNewton;
using Grammophone.Vectors;
using System.Runtime.Serialization;
using Grammophone.GenericContentModel;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Threading;
using System.Collections.Concurrent;

namespace Grammophone.CRF
{
	/// <summary>
	/// Linear Chain Conditional Random Field. Must be subclassed for use.
	/// </summary>
	/// <typeparam name="I">The type of the input as a whole.</typeparam>
	/// <typeparam name="T">The type of each tag in the output sequences.</typeparam>
	/// <remarks>
	/// <para>
	/// This implementation follows the notes of Elkan's
	/// "Log-Linear Models and Conditional Random Fields" (http://cseweb.ucsd.edu/~elkan/250B/loglinearCRFs.pdf)
	/// with own corrections included, combined with forward and backward vectors scaling for better arithmetic 
	/// robustness as described in Rabiner's 
	/// "A tutorial on hidden markov models and selected apllications in speech recognition"
	/// (http://www.cs.cornell.edu/Courses/cs4758/2012sp/materials/hmm_paper_rabiner.pdf) with 
	/// corrections provided by Rahimi's "An Erratum 
	/// for 'A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition'"
	/// (http://alumni.media.mit.edu/~rahimi/rabiner/rabiner-errata/rabiner-errata.html).
	/// </para>
	/// <para>
	/// WARNING: Some tutorials say Z = Σu α[n, u] or Z = Σu β[u, 1] where n is the position of the 
	/// last sequence element, NOT including the END tag, and 1 is the position of the first
	/// element of the sequence, NOT including the START tag.
	/// The above become wrong when we also say Ζ = Σu α[k, u] β[u, k].
	/// For k = 1 or k = n the latter relationship proves wrong the former ones,
	/// which means that they are different normalization constants.
	/// We should say Z = Σu α[|x|, u] instead, replacing n with |x| = n + 1 pointing at the END position,
	/// but at |x| = n + 1 there is only the END tag, so Z = α[|x|, END].
	/// Similarly, we should say Z = Σu β[u, 0] instead, but
	/// at position 0 there is only the START tag, so Z = β[START, 0].
	/// Due to the fact that we compute statistical figures using both forward α and backward β
	/// vectors, if we wanted to use positions 1 and n as edge cases and if we didn't scale the vectors as we do now,
	/// we would have to use the corrected formulae: 
	/// Z = Σu α[n, u] exp gn+1 [u, END]  or Z = Σu exp g1 [START, u] β[u, 1].
	/// </para>
	/// <para>
	/// Note: Charles Elkan's tutorial mixes the wrong version of Z from forward vectors
	/// and the correct version of Z from backward vectors.
	/// However, since we have scaled both forward α and backward β vectors according to Rabiner and Rahimi,
	/// the corresponding Z will always be Πk ck, see Rahimi's paragraph "Using α and β" for details.
	/// </para>
	/// </remarks>
	[Serializable]
	public abstract class LinearChainCRF<I, T>
	{
		#region Feature function definition

		/// <summary>
		/// Bigram vector feature function type for the Linear Chain Conditional Random Field.
		/// </summary>
		/// <param name="yim1">
		/// The output tag at position <paramref name="i"/> - 1.
		/// if <paramref name="i"/> is 0, the tag is START.
		/// </param>
		/// <param name="yi">
		/// The output tag at position <paramref name="i"/>.
		/// if <paramref name="i"/> is |x|, the tag is END.
		/// </param>
		/// <param name="x">
		/// The input.
		/// </param>
		/// <param name="i">
		/// The position in the output sequence. It can range from 0 to |y|, including.
		/// </param>
		/// <returns>
		/// Returns a vector of double values estimating the compatibility of the function arguments.
		/// Each member of the vector represents the output of a feature function.
		/// </returns>
		/// <remarks>
		/// The index i can range from 0 to |y|, including. In the edge cases i = 0 and i = |y|, where
		/// the <paramref name="yim1"/> is set to the START tag and the <paramref name="yi"/> is
		/// set to END tag correspondingly.
		/// </remarks>
		public delegate IVector BigramVectorFeatureFunction(T yim1, T yi, I x, int i);

		/// <summary>
		/// Unigram vector feature function type for the Linear Chain Conditional Random Field.
		/// </summary>
		/// <param name="yi">
		/// The output tag at position <paramref name="i"/>.
		/// if <paramref name="i"/> is |x|, the tag is END.
		/// </param>
		/// <param name="x">
		/// The input.
		/// </param>
		/// <param name="i">
		/// The position in the output sequence. It can range from 0 to |y|, including.
		/// </param>
		/// <returns>
		/// Returns a vector of double values estimating the compatibility of the function arguments.
		/// Each member of the vector represents the output of a feature function.
		/// </returns>
		/// <remarks>
		/// The index i can range from 0 to |y|, including. In the edge cases i = 0 and i = |y|.
		/// In the latter case, the <paramref name="yi"/> is
		/// set to END tag.
		/// </remarks>
		public delegate IVector UnigramVectorFeatureFunction(T yi, I x, int i);

		/// <summary>
		/// Provides <see cref="BigramVectorFeatureFunction"/>s, and gives a chance for caching
		/// the outcomes of the functions based on <see cref="Input"/>.
		/// </summary>
		public abstract class FeatureFunctionsProvider
		{
			#region Private fields

			private BigramVectorFeatureFunction bigramVectorFeatureFunction;

			private UnigramVectorFeatureFunction unigramVectorFeatureFunction;

			private int outputLength;

			#endregion

			#region Construction

			/// <summary>
			/// Create.
			/// </summary>
			/// <param name="input">The input.</param>
			public FeatureFunctionsProvider(I input)
			{
				if (input == null) throw new ArgumentNullException("input");

				this.Input = input;
				this.outputLength = -1;
			}

			#endregion

			#region Public properties

			/// <summary>
			/// The input sequence.
			/// </summary>
			public I Input { get; private set; }

			/// <summary>
			/// The bigram feature functions for an evaluation
			/// over the input sequence <see cref="Input"/>.
			/// </summary>
			public BigramVectorFeatureFunction BigramVectorFeatureFunction
			{
				get
				{
					if (bigramVectorFeatureFunction == null)
					{
						bigramVectorFeatureFunction = CreateBigramVectorFeatureFunction();
					}

					return bigramVectorFeatureFunction;
				}
			}

			/// <summary>
			/// The unigram feature functions for an evaluation
			/// over the input sequence <see cref="Input"/>.
			/// </summary>
			public UnigramVectorFeatureFunction UnigramVectorFeatureFunction
			{
				get
				{
					if (unigramVectorFeatureFunction == null)
					{
						unigramVectorFeatureFunction = CreateUnigramVectorFeatureFunction();
					}

					return unigramVectorFeatureFunction;
				}
			}

			/// <summary>
			/// The length of the output sequence based on the given <see cref="Input"/>,
			/// not including the START and END tags.
			/// </summary>
			public int OutputLength
			{
				get
				{
					if (outputLength < 0)
					{
						outputLength = GetOutputLength();
						if (outputLength < 0) throw new CRFException("Output length must be positive.");
					}

					return outputLength;
				}
			}

			#endregion

			#region Protected methods

			/// <summary>
			/// Implementations return the bigram feature functions for a evaluation
			/// over the <see cref="Input"/>.
			/// </summary>
			/// <remarks>
			/// It is expected that at all times, the method returns the same number of feature functions,
			/// since the given <see cref="Input"/> is constant.
			/// </remarks>
			protected abstract BigramVectorFeatureFunction CreateBigramVectorFeatureFunction();

			/// <summary>
			/// Implementations return the unigram feature functions for a evaluation
			/// over the <see cref="Input"/>.
			/// </summary>
			/// <remarks>
			/// It is expected that at all times, the method returns the same number of feature functions,
			/// since the given <see cref="Input"/> is constant.
			/// </remarks>
			protected abstract UnigramVectorFeatureFunction CreateUnigramVectorFeatureFunction();

			/// <summary>
			/// Implementations return the langth of the output based on the given input, 
			/// not including the START and END tags.
			/// </summary>
			/// <remarks>
			/// It is expected that at all times, the method returns the same length,
			/// since the given <see cref="Input"/> is constant.
			/// </remarks>
			protected abstract int GetOutputLength();

			#endregion
		}

		/// <summary>
		/// Specifies which <see cref="FeatureFunctionsProvider"/> to be used
		/// with the conditional random field.
		/// </summary>
		[Serializable]
		public abstract class FeatureFunctionsProviderFactory
		{
			#region Public properties

			/// <summary>
			/// Returns the number of feature functions.
			/// </summary>
			/// <remarks>
			/// Its is expected that this number never changes,
			/// meaning that the <see cref="FeatureFunctionsProvider.BigramVectorFeatureFunction"/> property
			/// of <see cref="FeatureFunctionsProvider"/> returned
			/// by method <see cref="GetProvider"/> has always the same value, equal.
			/// </remarks>
			public abstract int FeatureFunctionsCount
			{
				get;
			}

			#endregion

			#region Public methods

			/// <summary>
			/// Get the provider of <see cref="BigramVectorFeatureFunction"/>s.
			/// </summary>
			/// <param name="input">Optionally, take into account the input, for caching reasons.</param>
			/// <param name="scope">
			/// Gives a hint whether we are in training scope or not.
			/// It is an optional hint for caching precomputed structures required by feature functions.
			/// </param>
			/// <returns>Returns the provider of feature functions.</returns>
			/// <remarks>
			/// Its is expected that the length of <see cref="FeatureFunctionsProvider.BigramVectorFeatureFunction"/> property
			/// of <see cref="FeatureFunctionsProvider"/> returned
			/// by this method is always constant and equal to <see cref="FeatureFunctionsCount"/>.
			/// The method might neglect the optional <paramref name="input"/> 
			/// input sequence parameter, and return a static response.
			/// </remarks>
			public abstract FeatureFunctionsProvider GetProvider(I input, EvaluationScope scope);

			/// <summary>
			/// Get initial weights suggestion to start optimization algorithms.
			/// </summary>
			public abstract Vector GetInitialWeights();

			#endregion
		}

		#endregion

		#region Auxilliary types

		/// <summary>
		/// Represents a training pair of an input sequence and an output sequence.
		/// The sequences must be of the same length, of course.
		/// </summary>
		public struct TrainingPair
		{
			#region Public fields

			/// <summary>
			/// The input.
			/// </summary>
			public readonly I x;

			/// <summary>
			/// The output sequence.
			/// </summary>
			public readonly T[] y;

			#endregion

			#region Construction

			/// <summary>
			/// Create.
			/// </summary>
			/// <param name="x">The input.</param>
			/// <param name="y">The output sequence.</param>
			/// <remarks>
			/// The sequences must be of the same length, of course.
			/// </remarks>
			public TrainingPair(I x, T[] y)
			{
				if (x == null) throw new ArgumentNullException("x");
				if (y == null) throw new ArgumentNullException("y");

				this.x = x;
				this.y = y;
			}

			#endregion
		}

		#region Sequence evaluator base

		/// <summary>
		/// Represents the evaluation tool of a condition random field over an input sequence.
		/// </summary>
		public abstract class SequenceEvaluator
		{
			#region Private fields

			/// <summary>
			/// The special start tag, implied at position -1.
			/// </summary>
			protected T startTag;

			/// <summary>
			/// The special end tag, implied at position |x|.
			/// </summary>
			protected T endTag;

			/// <summary>
			/// The bigram feature functions.
			/// </summary>
			protected BigramVectorFeatureFunction f;

			/// <summary>
			/// The unigram feature functions.
			/// </summary>
			protected UnigramVectorFeatureFunction h;

			/// <summary>
			/// The current model parameters.
			/// </summary>
			protected Vector w;

			// The input.
			protected I x;

			private T[] y;

			private int outputLength;

			/// <summary>
			/// The cache for the computed partition function. If not already computed, it is -1.
			/// </summary>
			private double z = -1.0;

			private IVector featureFunctionExpectations;

			protected FeatureFunctionsProvider featureFunctionsProvider;

			#endregion

			#region Construction

			internal SequenceEvaluator(Vector w, LinearChainCRF<I, T> crf, FeatureFunctionsProvider featureFunctionsProvider)
			{
				if (w == null) throw new ArgumentNullException("w");
				if (crf == null) throw new ArgumentNullException("crf");
				if (featureFunctionsProvider == null) throw new ArgumentNullException("featureFunctionsProvider");

				this.startTag = crf.StartTag;
				this.endTag = crf.EndTag;
				this.f = featureFunctionsProvider.BigramVectorFeatureFunction;
				this.h = featureFunctionsProvider.UnigramVectorFeatureFunction;
				this.w = w;
				this.x = featureFunctionsProvider.Input;
				this.outputLength = featureFunctionsProvider.OutputLength;

				this.featureFunctionsProvider = featureFunctionsProvider;
			}

			#endregion

			#region Public properties

			/// <summary>
			/// Get the input.
			/// </summary>
			public I X
			{
				get
				{
					return this.x;
				}
			}

			/// <summary>
			/// Get the inferred sequence of tags for the given input.
			/// See pages 7 and 8 of Charles Elkan's
			/// Log Linear Models and Conditional Random Fields tutorial.
			/// If somehow a sequence is impossible for the given length, it returns null.
			/// </summary>
			/// <remarks>
			/// This is lazily evaluated. It is computed only the first time
			/// when a caller requests it, subsequent calls use the previously computed outcome.
			/// </remarks>
			public T[] Y
			{
				get
				{
					if (this.y == null) this.y = this.ComputeY();

					return this.y;
				}
			}

			/// <summary>
			/// The length of the output based on input <see cref="X"/>.
			/// </summary>
			public int OutputLength
			{
				get
				{
					return outputLength;
				}
			}

			/// <summary>
			/// Get the expectations of the feature functions according to the current configuration.
			/// See page 13 of Charles Elkan's
			/// Log Linear Models and Conditional Random Fields tutorial.
			/// </summary>
			/// <remarks>
			/// This is lazily evaluated. It is computed only the first time
			/// when a caller requests it, subsequent calls use the previously computed outcome.
			/// </remarks>
			public IVector FeatureFunctionsExpectations
			{
				get
				{
					if (this.featureFunctionExpectations == null) 
						this.featureFunctionExpectations = this.ComputeFeatureFunctionsExpectations();

					return this.featureFunctionExpectations;
				}
			}

			/// <summary>
			/// The partition function value.
			/// </summary>
			public double Z
			{
				get
				{
					if (this.z < 0.0) this.z = ComputePartitionFunction();
					return this.z;
				}
			}

			/// <summary>
			/// The feature functions provider used for this evaluation.
			/// </summary>
			public FeatureFunctionsProvider FeatureFunctionsProvider
			{
				get
				{
					return featureFunctionsProvider;
				}
			}

			#endregion

			#region Public methods

			/// <summary>
			/// Compute the vector of compound feature functions Fj values, each one consisting
			/// of the sum of fj feature function ofer the input sequence length.
			/// </summary>
			/// <param name="y">The output sequence.</param>
			/// <returns>
			/// The vector of the total feature functions, 
			/// each element containing the sum over sequence length 
			/// of linear chain feature functions fj.
			/// </returns>
			public IVector ComputeF(T[] y)
			{
				if (y == null) throw new ArgumentNullException("y");
				if (y.Length != outputLength) throw new ArgumentException("y must have the lenth specified by the input.", "y");

				IVector sum = f(startTag, y[0], x, 0).Clone();
				sum.AddInPlace(h(y[0], x, 0));

				for (int i = 1; i < y.Length; i++)
				{
					sum.AddInPlace(f(y[i - 1], y[i], x, i));
					sum.AddInPlace(h(y[i], x, i));
				}

				sum.AddInPlace(f(y[y.Length - 1], endTag, x, y.Length));
				sum.AddInPlace(h(endTag, x, y.Length));

				return sum;
			}

			/// <summary>
			/// Compute θlog p(y | x, w) / θw,
			/// which is the gradient, with respect to the vector of the
			/// weights, of the log-conditional-likelihood of the output sequence <paramref name="y"/>.
			/// </summary>
			/// <param name="y">The output sequence.</param>
			/// <returns>Returns the vector representing the gradient.</returns>
			/// <remarks>
			/// The output sequence <paramref name="y"/> must be of the same length with 
			/// inut sequence <see cref="X"/>.
			/// </remarks>
			public IVector ComputeLogConditionalLikelihoodGradient(T[] y)
			{
				var E = this.FeatureFunctionsExpectations;

				return this.ComputeF(y).Subtract(E);
			}

			/// <summary>
			/// Compute log p(y | x, w),
			/// which is the log-conditional-likelihood of the output sequence <paramref name="y"/>.
			/// </summary>
			/// <param name="y">The output sequence.</param>
			/// <returns>
			/// Returns the probability if the output sequence <paramref name="y"/>
			/// and the input sequence <see cref="X"/> have the same length, 
			/// else returns negative infinity.
			/// </returns>
			public double ComputeLogConditionalLikelihood(T[] y)
			{
				if (y == null) throw new ArgumentNullException("y");

				if (y.Length != outputLength) return Double.NegativeInfinity;

				return w * ComputeF(y) - Math.Log(Z);
			}

			#endregion

			#region Protected methods

			/// <summary>
			/// Compute the inferred sequence of tags for the given input.
			/// See pages 7 and 8 of Charles Elkan's
			/// Log Linear Models and Conditional Random Fields tutorial.
			/// If somehow a sequence is impossible for the given length, it returns null.
			/// </summary>
			protected abstract T[] ComputeY();

			protected abstract IVector ComputeFeatureFunctionsExpectations();

			protected abstract double ComputePartitionFunction();

			#endregion
		}

		#endregion

		#region Training options

		/// <summary>
		/// Abstract training options used for offline
		/// and online (stochastic gradient descent) training methods.
		/// </summary>
		[Serializable]
		public abstract class TrainingOptions
		{
			#region Private fields

			private double regularization = 1.0;

			#endregion

			#region Public properties

			/// <summary>
			/// The regularization constant C. Default is 1.0.
			/// </summary>
			/// <remarks>
			/// The regularized goal is:
			/// R(w) = -LCL(w) + C / 2len(w) * |w|^2.
			/// </remarks>
			public double Regularization
			{
				get
				{
					return regularization;
				}
				set
				{
					if (value < 0.0) throw new ArgumentException("Regularization constant must be non-negative");

					this.regularization = value;
				}
			}

			#endregion
		}

		/// <summary>
		/// Options for training the conditional random field offline.
		/// </summary>
		[Serializable]
		public class OfflineTrainingOptions : TrainingOptions
		{
			#region Private fields

			private Optimizer optimizer = new ConjugateGradientOptimizer();

			private GoalComputationFactory goalComputationFactory = new ParallelGoalComputationFactory();

			#endregion

			#region Public properties

			/// <summary>
			/// The optimizer to use.
			/// Default is <see cref="Grammophone.Optimization.QuasiNewton.ConjugateGradientOptimizer"/>.
			/// </summary>
			public Optimizer Optimizer
			{
				get
				{
					return optimizer;
				}
				set
				{
					if (value == null) throw new ArgumentNullException("value");

					optimizer = value;
				}
			}

			/// <summary>
			/// Factory contract for providing the goal computation 
			/// for the optimization of the training phase.
			/// Default is <see cref="ParallelGoalComputationFactory"/>.
			/// </summary>
			public GoalComputationFactory GoalComputationFactory
			{
				get
				{
					return goalComputationFactory;
				}
				set
				{
					if (value == null) throw new ArgumentNullException("value");
					goalComputationFactory = value;
				}
			}

			#endregion
		}

		/// <summary>
		/// Decay function for stochastic gradient descent.
		/// Must comply to the stochastic convergence conditions:
		/// It should decay to zero, but the sum must tend to infinity.
		/// </summary>
		[Serializable]
		public abstract class DecayFunction
		{
			public abstract double Evaluate(int iterationNumber);
		}

		/// <summary>
		/// The 1/n decay function.
		/// </summary>
		[Serializable]
		public class ReciprocalDecayFunction : DecayFunction
		{
			public override double Evaluate(int iterationNumber)
			{
				if (iterationNumber <= 0) return 1.0;

				return 1.0 / iterationNumber;
			}
		}

		/// <summary>
		/// Options for online training, ie stochastic gradient descent variants.
		/// </summary>
		[Serializable]
		public class OnlineTrainingOptions : TrainingOptions
		{
			#region Private fields

			private int maxIterationsCount = 1000;

			private Optimization.StochasticGradientDescent.Options optimizationOptions = new Optimization.StochasticGradientDescent.Options();

			private CancellationTokenSource cancellationTokenSource = new CancellationTokenSource();

			#endregion

			#region Public properties

			/// <summary>
			/// The maximum number of iterations.
			/// Default is 1000.
			/// </summary>
			public int MaxIterationsCount
			{
				get
				{
					return maxIterationsCount;
				}
				set
				{
					if (value < 0) throw new ArgumentException("value must be non negative.");
					maxIterationsCount = value;
				}
			}

			/// <summary>
			/// Options for stochastic gradient descent.
			/// </summary>
			public Optimization.StochasticGradientDescent.Options OptimizationOptions
			{
				get
				{
					return optimizationOptions;
				}
				set
				{
					if (value == null) throw new ArgumentNullException("value");
					optimizationOptions = value;
				}
			}

			/// <summary>
			/// A cancellation token source used to interrupt the training, if necessary.
			/// </summary>
			public CancellationTokenSource CancellationTokenSource
			{
				get
				{
					return cancellationTokenSource;
				}
				set
				{
					if (value == null) throw new ArgumentNullException("value");
					cancellationTokenSource = value;
				}
			}

			#endregion
		}

		#endregion

		#region Goal computation strategy system for off-line training

		/// <summary>
		/// Contract for computing the goal function and its gradient
		/// for off-line training.
		/// </summary>
		public abstract class GoalComputation
		{
			#region Private fields

			private int goalComputationsCount = 0;

			private int goalGradientComputationsCount = 0;

			#endregion

			#region Construction

			/// <summary>
			/// Initializes a new instance of the <see cref="GoalComputation"/> class.
			/// </summary>
			/// <param name="conditionalRandomField">The conditional random field.</param>
			/// <param name="trainingPairs">The training pairs.</param>
			/// <param name="trainingOptions">The options of training.</param>
			public GoalComputation(
				LinearChainCRF<I, T> conditionalRandomField,
				TrainingPair[] trainingPairs,
				OfflineTrainingOptions trainingOptions)
			{
				if (conditionalRandomField == null) throw new ArgumentNullException("conditionalRandomField");
				if (trainingPairs == null) throw new ArgumentNullException("trainingPairs");
				if (trainingOptions == null) throw new ArgumentNullException("trainingOptions");

				this.ConditionalRandomField = conditionalRandomField;
				this.TrainingPairs = trainingPairs;
				this.TrainingOptions = trainingOptions;
			}

			#endregion

			#region Protected properties

			/// <summary>
			/// The training set.
			/// </summary>
			protected TrainingPair[] TrainingPairs { get; private set; }

			/// <summary>
			/// The conditional random field.
			/// </summary>
			protected LinearChainCRF<I, T> ConditionalRandomField { get; private set; }

			/// <summary>
			/// The options specified for the training.
			/// </summary>
			protected OfflineTrainingOptions TrainingOptions { get; private set; }

			#endregion

			#region Public methods

			/// <summary>
			/// Compute the goal function.
			/// </summary>
			/// <param name="w">The value of the weights for which to compute the goal.</param>
			/// <returns>Returns the value of the goal function at <paramref name="w"/>.</returns>
			public abstract double ComputeGoal(Vector w);

			/// <summary>
			/// Compute the goal function gradient.
			/// </summary>
			/// <param name="w">The value of the weights for which to compute the goal gradient.</param>
			/// <returns>Returns the gradient of the goal function at <paramref name="w"/>.</returns>
			public abstract Vector ComputeGoalGradient(Vector w);

			/// <summary>
			/// Compute the regularized goal function.
			/// </summary>
			/// <param name="w">The value of the weights for which to compute the goal.</param>
			/// <returns>Returns the value of the goal function at <paramref name="w"/>.</returns>
			/// <remarks>
			/// This takes into account the <see cref="LinearChainCRF{I, T}.TrainingOptions.Regularization"/> field 
			/// of <see cref="TrainingOptions"/> property.
			/// </remarks>
			public virtual double ComputeRegularizedGoal(Vector w)
			{
				EnsureCompatibleWeightsVector(w);

				Trace.WriteLine(String.Format(
					"Computing regularized goal #{0}.",
					System.Threading.Interlocked.Increment(ref this.goalComputationsCount)));

				return 1e4 * (ComputeGoal(w) / TrainingPairs.Length + TrainingOptions.Regularization / (2.0 * w.Length) * w.Norm2);
			}

			/// <summary>
			/// Compute the regularized goal function gradient.
			/// </summary>
			/// <param name="w">The value of the weights for which to compute the goal gradient.</param>
			/// <returns>Returns the gradient of the goal function at <paramref name="w"/>.</returns>
			/// <remarks>
			/// This takes into account the <see cref="LinearChainCRF{I, T}.TrainingOptions.Regularization"/> field 
			/// of <see cref="TrainingOptions"/> property.
			/// </remarks>
			public virtual Vector ComputeRegularizedGoalGradient(Vector w)
			{
				EnsureCompatibleWeightsVector(w);

				Trace.WriteLine(String.Format(
					"Computing regularized goal gradient #{0}.",
					System.Threading.Interlocked.Increment(ref this.goalGradientComputationsCount)));

				return 1e4 * (ComputeGoalGradient(w) / TrainingPairs.Length + TrainingOptions.Regularization / w.Length * w);
			}

			#endregion 

			#region Protected methods

			/// <summary>
			/// Ensure that the vector <paramref name="w"/> has size 
			/// equal to the number of feature functions.
			/// </summary>
			/// <param name="w">The vector to test.</param>
			protected void EnsureCompatibleWeightsVector(Vector w)
			{
				if (w == null) throw new ArgumentNullException("w");

				if (w.Length != this.ConditionalRandomField.FunctionsProviderFactory.FeatureFunctionsCount)
					throw new ArgumentException("Vector w must have size equal to the number of feature functions.", "w");
			}

			#endregion
		}

		/// <summary>
		/// Factory contract for providing the goal computation 
		/// for the optimization of the training phase.
		/// Implementations of this class are expected to be set in <see cref="OfflineTrainingOptions"/>.
		/// </summary>
		[Serializable]
		public abstract class GoalComputationFactory
		{
			#region Private fields

			private int degreeOfParallelism = 0;

			#endregion

			#region Public properties

			/// <summary>
			/// The number of processor cores to use for the computation of the goal and its gradient.
			/// Zero means all available cores.
			/// Default is zero.
			/// </summary>
			public int DegreeOfParallelism
			{
				get
				{
					return degreeOfParallelism;
				}
				set
				{
					if (value < 0)
						throw new ArgumentException("The value must not be negative.");

					degreeOfParallelism = value;
				}
			}

			#endregion

			#region Public methods

			/// <summary>
			/// Get the class which computes the goal value and the goal gradient.
			/// </summary>
			/// <param name="conditionalRandomField">The conditional random field being trained.</param>
			/// <param name="trainingPairs">The training set.</param>
			/// <param name="trainingOptions">The options associated with training.</param>
			/// <returns>Returns the class for computing the goal and its gradient.</returns>
			public abstract GoalComputation GetGoalComputation(
				LinearChainCRF<I, T> conditionalRandomField,
				TrainingPair[] trainingPairs,
				OfflineTrainingOptions trainingOptions);

			#endregion
		}

		/// <summary>
		/// Parallel computation of the goal function and its gradient
		/// for off-line training.
		/// </summary>
		public class ParallelGoalComputation : GoalComputation
		{
			#region Private fields

			private int degreeOfParallelism;

			#endregion

			#region Construction

			/// <summary>
			/// Create.
			/// </summary>
			/// <param name="conditionalRandomField">The conditional random field.</param>
			/// <param name="trainingPairs">The training pairs.</param>
			/// <param name="trainingOptions">The options of training.</param>
			/// <param name="factory">The factory which created this object.</param>
			internal ParallelGoalComputation(
				LinearChainCRF<I, T> conditionalRandomField,
				TrainingPair[] trainingPairs,
				OfflineTrainingOptions trainingOptions,
				ParallelGoalComputationFactory factory)
				: base(conditionalRandomField, trainingPairs, trainingOptions)
			{
				if (factory == null) throw new ArgumentNullException("factory");

				this.degreeOfParallelism = factory.DegreeOfParallelism;

				if (this.degreeOfParallelism == 0)
				{
					this.degreeOfParallelism = Environment.ProcessorCount;
				}
			}

			#endregion

			#region Public methods

			public override double ComputeGoal(Vector w)
			{
				EnsureCompatibleWeightsVector(w);

				var trainingPairsPartitioner = Partitioner.Create(this.TrainingPairs, true);

				var likelihoods = from trainingPair in trainingPairsPartitioner.AsParallel().WithDegreeOfParallelism(degreeOfParallelism)
													select
													this.ConditionalRandomField.GetSequenceEvaluator(w, trainingPair.x, EvaluationScope.Training)
													.ComputeLogConditionalLikelihood(trainingPair.y);

				return -likelihoods.Sum();
			}

			public override Vector ComputeGoalGradient(Vector w)
			{
				EnsureCompatibleWeightsVector(w);

				var g = new Vector(w.Length);

				var trainingPairsPartitioner = Partitioner.Create(this.TrainingPairs, true);

				var gradients = from trainingPair in trainingPairsPartitioner.AsParallel().WithDegreeOfParallelism(degreeOfParallelism)
												select
												this.ConditionalRandomField.GetSequenceEvaluator(w, trainingPair.x, EvaluationScope.Training)
												.ComputeLogConditionalLikelihoodGradient(trainingPair.y);

				gradients.Sum(g);

				return -g;
			}

			#endregion
		}

		/// <summary>
		/// Factory providing a parallel implementation for computing the goal
		/// and its gradient.
		/// </summary>
		public class ParallelGoalComputationFactory : GoalComputationFactory
		{
			#region Public methods

			public override GoalComputation GetGoalComputation(
				LinearChainCRF<I, T> conditionalRandomField,
				TrainingPair[] trainingPairs,
				OfflineTrainingOptions trainingOptions
				)
			{
				return new ParallelGoalComputation(conditionalRandomField, trainingPairs, trainingOptions, this);
			}

			#endregion
		}

		#endregion

		#endregion

		#region Private fields

		/// <summary>
		/// The factory for <see cref="FeatureFunctionsProvider"/>.
		/// </summary>
		private FeatureFunctionsProviderFactory functionsProviderFactory;

		/// <summary>
		/// The model parameters vector.
		/// </summary>
		private Vector w;

		/// <summary>
		/// The special start tag, implied at position -1.
		/// </summary>
		private T startTag;

		/// <summary>
		/// The special end tag, implied at position |x|.
		/// </summary>
		private T endTag;

		#endregion

		#region Construction

		/// <summary>
		/// Create.
		/// </summary>
		/// <param name="startTag">The special tag used to mark the start of a sentence, implied at index -1. Does not belong to the normal tags.</param>
		/// <param name="endTag">The special tag used to mark the end of a sentence, implied at index |x|. Does not belong to the normal tags.</param>
		/// <param name="featureFunctionsProviderFactory">The provider of feature functions.</param>
		public LinearChainCRF(T startTag, T endTag, FeatureFunctionsProviderFactory featureFunctionsProviderFactory)
		{
			if (startTag == null) throw new ArgumentNullException("startTag");
			if (endTag == null) throw new ArgumentNullException("endTag");
			if (featureFunctionsProviderFactory == null) throw new ArgumentNullException("featureFunctionsProviderFactory");

			this.startTag = startTag;
			this.endTag = endTag;
			this.functionsProviderFactory = featureFunctionsProviderFactory;
		}

		#endregion

		#region Public properties

		/// <summary>
		/// True if this conditional random field is trained.
		/// </summary>
		public bool IsTrained
		{
			get
			{
				return this.w != null;
			}
		}

		/// <summary>
		/// If <see cref="IsTrained"/> is true, it returns the model parameters of the conditional random field,
		/// else returns null. Intended for read-only access.
		/// </summary>
		public Vector Weights
		{
			get
			{
				return w;
			}
		}

		/// <summary>
		/// The factory for providing feature functions.
		/// </summary>
		public FeatureFunctionsProviderFactory FunctionsProviderFactory
		{
			get
			{
				return functionsProviderFactory;
			}
		}

		/// <summary>
		/// The special tag used to mark the start of a sentence, implied at index -1. Does not belong to the normal tags.
		/// </summary>
		public T StartTag
		{
			get
			{
				return startTag;
			}
		}

		/// <summary>
		/// The special tag used to mark the end of a sentence, implied at index |x|. Does not belong to the normal tags.
		/// </summary>
		public T EndTag
		{
			get
			{
				return endTag;
			}
		}

		#endregion

		#region Public methods

		/// <summary>
		/// Returns an evaluator of the conditional random field over
		/// an input sequence.
		/// The conditional random field must be trained or loaded with trained weights,
		/// else an ApplicationException is thrown.
		/// </summary>
		/// <param name="x">The input sequence.</param>
		/// <returns>
		/// Returns an evaluator of the conditional random field for 
		/// the given input <paramref name="x"/>.
		/// </returns>
		/// <remarks>
		/// See <see cref="IsTrained"/> property.
		/// </remarks>
		public SequenceEvaluator GetSequenceEvaluator(I x)
		{
			if (!this.IsTrained)
				throw new ApplicationException(
					"The Conditional Random Field is not trained or loaded with trained weights.");

			return GetSequenceEvaluator(w, x, EvaluationScope.Running);
		}

		/// <summary>
		/// Train the conditional random field off-line, that is, all the members of the
		/// training set are assumed to be present beforehand in <paramref name="trainingPairs"/>.
		/// </summary>
		/// <param name="trainingPairs">The training set.</param>
		/// <param name="trainingOptions">The options for training.</param>
		public void OfflineTrain(TrainingPair[] trainingPairs, OfflineTrainingOptions trainingOptions)
		{
			if (trainingPairs == null) throw new ArgumentNullException("trainingPairs");
			if (trainingOptions == null) throw new ArgumentNullException("trainingOptions");

			this.w = null; // Mark as untrained.

			// Initialization of weights.
			var w = this.FunctionsProviderFactory.GetInitialWeights();

			var goalComputation = 
				trainingOptions.GoalComputationFactory.GetGoalComputation(this, trainingPairs, trainingOptions);

			this.w = trainingOptions.Optimizer.Minimize(goalComputation.ComputeRegularizedGoal, goalComputation.ComputeRegularizedGoalGradient, w); 
		}

		/// <summary>
		/// Train the conditional random field on-line from a sequence of incoming training pairs using L2 regularization.
		/// The sequence can be infinite, but the training options specify the maximum items to take from it.
		/// </summary>
		/// <param name="trainingPairsSequence">The sequence of training pairs, possibly infinite.</param>
		/// <param name="trainingOptions">The options for training.</param>
		/// <param name="degreeOfParallelism">
		/// The degree of parallelism or zero for all available processor cores.
		/// A value above tha available cores will be ignores.
		/// </param>
		public void OnlineTrain(IEnumerable<TrainingPair> trainingPairsSequence, OnlineTrainingOptions trainingOptions, int degreeOfParallelism)
		{
			if (trainingPairsSequence == null) throw new ArgumentNullException("trainingPairsSequence");
			if (trainingOptions == null) throw new ArgumentNullException("trainingOptions");
			if (degreeOfParallelism < 0) throw new ArgumentException("degreeOfParallelism must not be negative.");

			if (degreeOfParallelism == 0 || degreeOfParallelism > Environment.ProcessorCount) degreeOfParallelism = Environment.ProcessorCount;

			this.w = Optimization.StochasticGradientDescent.ParallelMinimize(
				trainingPairsSequence,
				(trainingPair, w) => trainingOptions.Regularization * w - GetSequenceEvaluator(w, trainingPair.x, EvaluationScope.Training).ComputeLogConditionalLikelihoodGradient(trainingPair.y),
				this.FunctionsProviderFactory.GetInitialWeights(),
				trainingOptions.OptimizationOptions,
				trainingOptions.CancellationTokenSource.Token,
				degreeOfParallelism);
		}

		#endregion

		#region Protected methods

		/// <summary>
		/// Specifies the evaluator implementation for the conditional random field over
		/// an input sequence.
		/// </summary>
		/// <param name="weights">The model weights.</param>
		/// <param name="featureFunctionsProvider">The feature functions provider obtained by the CRF's <see cref="FeatureFunctionsProviderFactory"/>.</param>
		/// <returns>Returns an implementation instance of the <see cref="SequenceEvaluator"/> class.</returns>
		/// <remarks>
		/// The <see cref="FeatureFunctionsProvider"/> also contains the <see cref="FeatureFunctionsProvider.Input"/> field.
		/// </remarks>
		protected abstract SequenceEvaluator GetSequenceEvaluator(Vector weights, FeatureFunctionsProvider featureFunctionsProvider);

		/// <summary>
		/// Returns an evaluator of the conditional random field over
		/// an input sequence.
		/// The conditional random field must be trained or loaded with trained weights,
		/// else an ApplicationException is thrown.
		/// </summary>
		/// <param name="weights">The model weights.</param>
		/// <param name="x">The input sequence.</param>
		/// <param name="scope">
		/// Gives a hint whether we are in training scope or not.
		/// It is an optional hint for caching precomputed structures required by feature functions.
		/// </param>
		/// <returns>
		/// Returns an evaluator of the conditional random field for 
		/// the given input <paramref name="x"/>.
		/// </returns>
		/// <remarks>
		/// See <see cref="IsTrained"/> property.
		/// </remarks>
		protected SequenceEvaluator GetSequenceEvaluator(Vector weights, I x, EvaluationScope scope)
		{
			if (weights == null) throw new ArgumentNullException("weights");
			if (x == null) throw new ArgumentNullException("x");

			var featureFunctionsProvider = this.FunctionsProviderFactory.GetProvider(x, scope);

			return GetSequenceEvaluator(weights, featureFunctionsProvider);
		}

		#endregion
	}
}
