using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using Grammophone.GenericContentModel;
using Grammophone.Vectors;
using Grammophone.Caching;

namespace Grammophone.CRF
{
	/// <summary>
	/// Linear Chain Conditional Random Field. Must be subclassed for use.
	/// Constrains bi-gram combinations.
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
	public abstract class ConstrainedLinearChainCRF<I, T> : LinearChainCRF<I, T>, IDeserializationCallback
		where T : IEquatable<T>
	{
		#region Auxilliary classes

		/// <summary>
		/// Represents the evaluation tool of a condition random field over an input sequence.
		/// </summary>
		private class ConstrainedSequenceEvaluator : SequenceEvaluator
		{
			#region Private fields

			/// <summary>
			/// All the possible pairs of consecutive tags, including the special START and END tags.
			/// </summary>
			private IReadOnlyBiGramSet<T> tagBiGrams;

			/// <summary>
			/// Cache holding the allowed tags for a given sequence length, 
			/// as implied by <see cref="tagBiGrams"/>.
			/// </summary>
			private MRUCache<int, ISet<T>[]> allowedTagsByIndexCache;

			// private double[][,] g;
			private Dictionary<Tuple<T, T>, double>[] g;

			// private double[][,] expG;
			private Dictionary<Tuple<T, T>, double>[] expG;

			// private double[,] u;
			private Dictionary<T, double>[] u;

			private Dictionary<T, double>[] forwardVector;

			private Dictionary<T, double>[] backwardVector;

			/// <summary>
			/// This is the array of scales for the forward and backward vectors.
			/// This gets computed after the <see cref="ComputeForwardVector"/> method invokation.
			/// </summary>
			/// <remarks>
			/// It is implied that c[-1] = 1 and c[|x|] = 1.
			/// </remarks>
			private double[] c;

			#endregion

			#region Construction

			public ConstrainedSequenceEvaluator(Vector w, ConstrainedLinearChainCRF<I, T> crf, FeatureFunctionsProvider featureFunctionsProvider)
				: base(w, crf, featureFunctionsProvider)
			{
				this.tagBiGrams = crf.TagBiGrams;
				this.allowedTagsByIndexCache = crf.AllowedTagsByIndexCache;
			}

			#endregion

			#region Public properties

			/// <summary>
			/// Get the g matrix of matrices as specified in page 7 of Charles Elkan's
			/// Log Linear Models and Conditional Random Fields tutorial.
			/// </summary>
			/// <remarks>
			/// This is lazily evaluated. It is computed only the first time
			/// when a caller requests it, subsequent calls use the previously computed outcome.
			/// </remarks>
			public Dictionary<Tuple<T, T>, double>[] G
			{
				get
				{
					if (this.g == null) this.g = this.ComputeG();

					return this.g;
				}
			}

			/// <summary>
			/// Get the matrix of the exponentials of the elements of the g 
			/// matrix of matrices as specified in page 7 of Charles Elkan's
			/// Log Linear Models and Conditional Random Fields tutorial.
			/// This is commonly used for the computation of the forward and backward vectors
			/// and the expectation of the feature functions as seen in pages 11, 12, 13.
			/// </summary>
			/// <remarks>
			/// This is lazily evaluated. It is computed only the first time
			/// when a caller requests it, subsequent calls use the previously computed outcome.
			/// </remarks>
			public Dictionary<Tuple<T, T>, double>[] ExpG
			{
				get
				{
					if (this.expG == null) this.expG = this.ComputeExpG();

					return this.expG;
				}
			}

			/// <summary>
			/// Get the matrix U[k, v] of scores of best sequence from position 0 to position k
			/// where the tag number in position k is equal to tagSet[v].
			/// See pages 7 and 8 of Charles Elkan's
			/// Log Linear Models and Conditional Random Fields tutorial.
			/// </summary>
			/// <remarks>
			/// This is lazily evaluated. It is computed only the first time
			/// when a caller requests it, subsequent calls use the previously computed outcome.
			/// </remarks>
			public Dictionary<T, double>[] U
			{
				get
				{
					if (this.u == null) this.u = this.ComputeU();

					return this.u;
				}
			}

			/// <summary>
			/// Get the forward vector. This is typically used in the computation
			/// of the expectation of the feature functions and other statistical
			/// figures.
			/// See pages 11, 12, 13 of Charles Elkan's
			/// Log Linear Models and Conditional Random Fields tutorial.
			/// However, it is scaled according to 
			/// "A tutorial on hidden markov models and selected apllications in speech recognition"
			/// (http://www.cs.cornell.edu/Courses/cs4758/2012sp/materials/hmm_paper_rabiner.pdf) with 
			/// corrections provided by Rahimi's "An Erratum 
			/// for 'A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition'"
			/// (http://alumni.media.mit.edu/~rahimi/rabiner/rabiner-errata/rabiner-errata.html).
			/// </summary>
			/// <remarks>
			/// This is lazily evaluated. It is computed only the first time
			/// when a caller requests it, subsequent calls use the previously computed outcome.
			/// </remarks>
			public Dictionary<T, double>[] ForwardVector
			{
				get
				{
					if (forwardVector == null) this.forwardVector = this.ComputeForwardVector();

					return this.forwardVector;
				}
			}

			/// <summary>
			/// Get the backward vector. This is typically used in the computation
			/// of the expectation of the feature functions and other statistical
			/// figures.
			/// See pages 11, 12, 13 of Charles Elkan's
			/// Log Linear Models and Conditional Random Fields tutorial.
			/// However, it is scaled according to 
			/// "A tutorial on hidden markov models and selected apllications in speech recognition"
			/// (http://www.cs.cornell.edu/Courses/cs4758/2012sp/materials/hmm_paper_rabiner.pdf) with 
			/// corrections provided by Rahimi's "An Erratum 
			/// for 'A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition'"
			/// (http://alumni.media.mit.edu/~rahimi/rabiner/rabiner-errata/rabiner-errata.html).
			/// </summary>
			/// <remarks>
			/// This is lazily evaluated. It is computed only the first time
			/// when a caller requests it, subsequent calls use the previously computed outcome.
			/// </remarks>
			public Dictionary<T, double>[] BackwardVector
			{
				get
				{
					if (this.backwardVector == null) this.backwardVector = this.ComputeBackwardVector();

					return this.backwardVector;
				}
			}

			#endregion

			#region Protected methods

			protected override T[] ComputeY()
			{
				int sequenceCount = this.OutputLength;

				T[] y = new T[sequenceCount];

				var u = this.U;

				double max;

				var g = this.G;

				// WARNING: Our base case is different from most tutorials which specify:
				// yn = max_v u[k, v]
				// We use the following for consistency instead, because, throughout our whole approach,
				// our feature functions span up to the END position, including.
				// yn = max_v u[k, v] + gn[v, END];
				// So, We can go ahead directly to the recursion which has the same form as above.

				// Recursion.

				T ykm1 = startTag;

				T yk = endTag;

				for (int km1 = y.Length - 1; km1 >= 0; km1--)
				{
					var gk = g[km1 + 1];

					var ukm1 = u[km1];

					max = double.NegativeInfinity;

					var biGrams = tagBiGrams.GetBySecond(yk);

					foreach (var biGram in biGrams)
					{
						var ykm1Candidate = biGram.Item1;

						double ukm1_ykm1;

						if (!ukm1.TryGetValue(ykm1Candidate, out ukm1_ykm1)) continue;

						double candidate = ukm1_ykm1 + gk[biGram];

						if (candidate > max)
						{
							max = candidate;
							ykm1 = ykm1Candidate;
						}
					}

					if (max == Double.NegativeInfinity) return null;

					y[km1] = ykm1;

					yk = ykm1;
				}

				return y;
			}

			// Use the method described in the paragraph "Using α and β" of 
			// "An Erratum for 'A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition'"
			// (http://alumni.media.mit.edu/~rahimi/rabiner/rabiner-errata/rabiner-errata.html).
			// to compute Z from the scaling constants.
			protected override double ComputePartitionFunction()
			{
				if (this.OutputLength == 0) return 1.0;

				// This forces computing the ck scaling weights, if not computed already.
				var β = this.BackwardVector;

				double c0 = c[0];

				if (c0 == 0.0) return 1.0;

				var expG0 = this.ExpG[0];

				var β0 = β[0];

				double zu = 0.0;

				foreach (var entry in β0)
				{
					var y0 = entry.Key;

					zu += entry.Value * expG0[new Tuple<T, T>(startTag, y0)];
				}

				return zu / c0;
			}

			protected override IVector ComputeFeatureFunctionsExpectations()
			{
				var α = this.ForwardVector;
				var β = this.BackwardVector;
				var expG = this.ExpG;

				var E = new Vector(w.Length);

				var allowedTagsByIndex = allowedTagsByIndexCache.Get(this.OutputLength);

				// i = 0
				{
					var expG0 = expG[0];

					var tagsIn0 = allowedTagsByIndex[0];

					var startBiGrams = new Tuple<T, T>[tagsIn0.Count]; // tagBiGrams.GetByFirst(startTag);

					var β0 = β[0];

					foreach (var y0 in tagsIn0)
					{
						double β0_y0;

						if (!β0.TryGetValue(y0, out β0_y0)) continue;

						var biGram = new Tuple<T, T>(startTag, y0);

						double factor = expG0[biGram] * β0_y0;

						E.AddInPlace((f(startTag, y0, x, 0).Add(h(y0, x, 0))).Scale(factor));
					}
				}

				for (int i = 1; i < this.OutputLength; i++)
				{
					var expGi = expG[i];

					var αim1 = α[i - 1];
					var βi = β[i];

					var tagsInI = allowedTagsByIndex[i];

					foreach (T yi in tagsInI)
					{
						double βi_yi;

						if (!βi.TryGetValue(yi, out βi_yi)) continue;

						var previousBiGrams = tagBiGrams.GetBySecond(yi);

						double totalFactor = 0.0;

						foreach (var biGram in previousBiGrams)
						{
							T yim1 = biGram.Item1;

							double αim1_yim1;

							if (!αim1.TryGetValue(yim1, out αim1_yim1)) continue;

							double factor = αim1_yim1 * expGi[biGram] * βi_yi;

							totalFactor += factor;

							E.AddInPlace(f(yim1, yi, x, i).Scale(factor));
						}

						if (totalFactor != 0.0)
						{
							E.AddInPlace(h(yi, x, i).Scale(totalFactor));
						}
					}

				}

				// i = n
				{
					int n = this.OutputLength;
					int nm1 = this.OutputLength - 1;

					var expGn = expG[n];

					var tagsInNm1 = allowedTagsByIndex[n - 1];

					var αnm1 = α[nm1];

					double totalFactor = 0.0;

					foreach (var ynm1 in tagsInNm1)
					{
						double αnm1_ynm1;

						if (!αnm1.TryGetValue(ynm1, out αnm1_ynm1)) continue;

						var biGram = new Tuple<T, T>(ynm1, endTag);

						double factor = αnm1_ynm1 * expGn[biGram];

						E.AddInPlace(f(ynm1, endTag, x, n).Scale(factor));
					}

					if (totalFactor != 0.0)
					{
						E.AddInPlace(h(endTag, x, n).Scale(totalFactor));
					}
				}

				return E.DivideInPlace(this.Z);
			}

			#endregion

			#region Private methods

			private Dictionary<Tuple<T, T>, double>[] ComputeG()
			{
				int sequenceCount = this.OutputLength;

				var g = new Dictionary<Tuple<T, T>, double>[sequenceCount + 1];

				var allowedTagsByIndex = allowedTagsByIndexCache.Get(sequenceCount);

				// Compute special start case i = 0.
				{
					var gi = new Dictionary<Tuple<T, T>, double>();

					g[0] = gi;

					var tagsIn0 = allowedTagsByIndex[0];

					foreach (var y0 in tagsIn0)
					{
						var startBiGram = new Tuple<T, T>(startTag, y0);

						gi[startBiGram] = w * (f(startTag, y0, x, 0).Add(h(y0, x, 0)));
					}
				}

				// Compute for mid part of sequence x.
				for (int i = 1; i < sequenceCount; i++)
				{
					var gi = new Dictionary<Tuple<T, T>, double>();
					g[i] = gi;

					/* No need to compute all bi-grams. Use the tagsByIndex to infer the valid bi-grams at this index of the sequence. */

					var tagsInIm1 = allowedTagsByIndex[i - 1];

					var tagsInI = allowedTagsByIndex[i];

					foreach (var yi in tagsInI)
					{
						double unigramComponent = w * h(yi, x, i);

						var previousBiGrams = tagBiGrams.GetBySecond(yi);

						foreach (var biGram in previousBiGrams)
						{
							T yim1 = biGram.Item1;

							if (!tagsInIm1.Contains(yim1)) continue;

							gi[biGram] = w * f(yim1, yi, x, 0) + unigramComponent;
						}
					}

				}

				// Compute special end case i = |x|.
				{
					var gi = new Dictionary<Tuple<T, T>, double>();

					g[sequenceCount] = gi;

					var tagsInNm1 = allowedTagsByIndex[sequenceCount - 1];

					double unigramComponent = w * h(endTag, x, sequenceCount);

					foreach (var ynm1 in tagsInNm1)
					{
						var endBiGram = new Tuple<T, T>(ynm1, endTag);

						gi[endBiGram] = w * f(ynm1, endTag, x, sequenceCount) + unigramComponent;
					}

				}

				return g;
			}

			private Dictionary<Tuple<T, T>, double>[] ComputeExpG()
			{
				var g = this.G;

				var expG = new Dictionary<Tuple<T, T>, double>[g.Length];

				for (int i = 0; i < g.Length; i++)
				{
					var gi = g[i];

					var expGi = new Dictionary<Tuple<T, T>, double>(gi.Count);

					expG[i] = expGi;

					foreach (var entry in gi)
					{
						expGi[entry.Key] = Math.Exp(entry.Value);
					}

				}

				return expG;
			}

			private Dictionary<T, double>[] ComputeU()
			{
				int sequenceCount = this.OutputLength;

				var u = new Dictionary<T, double>[sequenceCount + 1];

				var g = this.G;

				var allowedTagsByIndex = allowedTagsByIndexCache.Get(sequenceCount);

				// Base case k = 0
				var g0 = g[0];

				var u0 = new Dictionary<T, double>();

				var tagsIn0 = allowedTagsByIndex[0];

				foreach (var y0 in tagsIn0)
				{
					var startBiGram = new Tuple<T, T>(startTag, y0);

					u0[y0] = g0[startBiGram];
				}

				u[0] = u0;

				// Recursion

				var ukm1 = u0;

				for (int k = 1; k <= sequenceCount; k++)
				{
					var gk = g[k];

					var tagsInK = allowedTagsByIndex[k];

					var uk = new Dictionary<T, double>();

					foreach (var v in tagsInK)
					{
						var biGrams = tagBiGrams.GetBySecond(v);

						double max = Double.NegativeInfinity;

						foreach (var biGram in biGrams)
						{
							T y = biGram.Item1;

							double ukm1_y;

							if (!ukm1.TryGetValue(y, out ukm1_y)) continue;

							double candidate = ukm1_y + gk[biGram];

							if (candidate > max) max = candidate;
						}

						uk[v] = max;
					}

					u[k] = uk;
					ukm1 = uk;

				}

				return u;
			}

			/// <summary>
			/// When a forward or backward vector reaches a stage of all zeros, 
			/// then it means that a sequence of the given length is infeasible, and the algorithm cannot continue.
			/// The probabilities must be all zero. 
			/// This method is used in such cases, and it returns an empty vector and sets all c coefficients to zero.
			/// </summary>
			private Dictionary<T, double>[] GetZeroVector()
			{
				int sequenceCount = this.OutputLength;

				this.c = new double[sequenceCount];

				var vector = new Dictionary<T, double>[sequenceCount];

				for (int i = 0; i < vector.Length; i++)
				{
					vector[i] = new Dictionary<T, double>();
				}

				return vector;
			}

			private Dictionary<T, double>[] ComputeForwardVector()
			{
				int sequenceCount = this.OutputLength;

				var α = new Dictionary<T, double>[sequenceCount];

				this.c = new double[sequenceCount];

				var expG = this.ExpG;

				var allowedTagsByIndex = allowedTagsByIndexCache.Get(sequenceCount);

				// This will hold the the normalizer for each level.
				double ck;
				// This will hold the inverse of the normalizer for each level.
				double invck;

				// Base case.

				invck = 0.0;

				var expG0 = expG[0];

				var tagsIn0 = allowedTagsByIndex[0];

				var α0 = new Dictionary<T, double>(tagsIn0.Count);
				α[0] = α0;

				foreach (var y0 in tagsIn0)
				{
					var biGram = new Tuple<T, T>(startTag, y0);

					double unnormalized = expG0[biGram];

					α0[y0] = unnormalized;

					invck += unnormalized;
				}

				if (invck == 0.0) return GetZeroVector();

				ck = 1 / invck;
				c[0] = ck;

				// Rescale.

				var α0entries = α0.ToArray();

				foreach (var entry in α0entries)
				{
					α0[entry.Key] = ck * entry.Value;
				}

				// Recursion.

				var αkm1 = α0;

				for (int k = 1; k < sequenceCount; k++)
				{
					var expGk = expG[k];

					invck = 0.0;

					var tagsInK = allowedTagsByIndex[k];

					var αk = new Dictionary<T, double>(tagsInK.Count);
					α[k] = αk;

					foreach (var v in tagsInK)
					{
						var previousBiGrams = tagBiGrams.GetBySecond(v);

						double sum = 0.0;

						foreach (var biGram in previousBiGrams)
						{
							double αkm1_u;

							if (!αkm1.TryGetValue(biGram.Item1, out αkm1_u)) continue;

							sum += αkm1_u * expGk[biGram];
						}

						αk[v] = sum;

						invck += sum;
					}

					if (invck == 0.0) return GetZeroVector();

					ck = 1 / invck;
					c[k] = ck;

					// Rescale.

					var αkentries = αk.ToArray();

					foreach (var entry in αkentries)
					{
						αk[entry.Key] = ck * entry.Value;
					}

					αkm1 = αk;
				}

				return α;
			}

			private Dictionary<T, double>[] ComputeBackwardVector()
			{
				// Force computing the forward vector first, if not already computed,
				// in order to create the c[k] normalization weights.
				var α = this.ForwardVector;

				int sequenceCount = this.OutputLength;

				var β = new Dictionary<T, double>[sequenceCount];

				var expG = this.ExpG;

				var allowedTagsByIndex = allowedTagsByIndexCache.Get(sequenceCount);

				double ck;

				// Base case.

				var expGn = expG[sequenceCount];

				ck = c[sequenceCount - 1];

				var tagsInNm1 = allowedTagsByIndex[sequenceCount - 1];

				var βnm1 = new Dictionary<T, double>(tagsInNm1.Count);
				β[sequenceCount - 1] = βnm1;

				var αnm1 = α[sequenceCount - 1];

				foreach (var ynm1 in tagsInNm1)
				{
					var biGram = new Tuple<T, T>(ynm1, endTag);

					βnm1[ynm1] = expGn[biGram] * ck;
				}

				// Recursion.

				var βkp1 = βnm1;

				for (int k = sequenceCount - 2; k >= 0; k--)
				{
					int kp1 = k + 1;

					var expGkp1 = expG[kp1];

					ck = c[k];

					var tagsInK = allowedTagsByIndex[k];

					var βk = new Dictionary<T, double>(tagsInK.Count);
					β[k] = βk;

					foreach (var u in tagsInK)
					{
						var nextBiGrams = tagBiGrams.GetByFirst(u);

						double sum = 0.0;

						foreach (var biGram in nextBiGrams)
						{
							var v = biGram.Item2;

							double βkp1_v;

							if (!βkp1.TryGetValue(v, out βkp1_v)) continue;

							sum += expGkp1[biGram] * βkp1_v;
						}

						βk[u] = sum * ck;
					}

					βkp1 = βk;
				}

				return β;
			}

			#endregion
		}

		#endregion

		#region Private fields

		[NonSerialized]
		private LazySequence<ISet<T>> forwardTagsByIndex;

		[NonSerialized]
		private LazySequence<ISet<T>> backwardTagsByReverseIndex;

		[NonSerialized]
		private MRUCache<int, ISet<T>[]> allowedTagsByIndexCache;

		#endregion

		#region Construction

		/// <summary>
		/// Create.
		/// </summary>
		/// <param name="tagBiGrams">All the possible pairs of consecutive tags, including the special START and END tags.</param>
		/// <param name="startTag">The special tag used to mark the start of a sentence, implied at index -1. Does not belong to the normal tags.</param>
		/// <param name="endTag">The special tag used to mark the end of a sentence, implied at index |x|. Does not belong to the normal tags.</param>
		/// <param name="featureFunctionsProviderFactory">The provider of feature functions.</param>
		public ConstrainedLinearChainCRF(IReadOnlyBiGramSet<T> tagBiGrams, T startTag, T endTag, FeatureFunctionsProviderFactory featureFunctionsProviderFactory)
			: base(startTag, endTag, featureFunctionsProviderFactory)
		{
			if (tagBiGrams == null) throw new ArgumentNullException("tagBiGrams");

			this.TagBiGrams = tagBiGrams;

			InitializeTagSetSequences();
		}

		/// <summary>
		/// Create.
		/// </summary>
		/// <param name="tagBiGrams">All the possible pairs of consecutive tags, including the special START and END tags.</param>
		/// <param name="startTag">The special tag used to mark the start of a sentence, implied at index -1. Does not belong to the normal tags.</param>
		/// <param name="endTag">The special tag used to mark the end of a sentence, implied at index |x|. Does not belong to the normal tags.</param>
		/// <param name="featureFunctionsProviderFactory">The provider of feature functions.</param>
		public ConstrainedLinearChainCRF(IEnumerable<Tuple<T, T>> tagBiGrams, T startTag, T endTag, FeatureFunctionsProviderFactory featureFunctionsProviderFactory)
			: this(new ReadOnlyBiGramSet<T>(tagBiGrams), startTag, endTag, featureFunctionsProviderFactory)
		{
		}

		#endregion

		#region Public properties

		/// <summary>
		/// All the possible pairs of consecutive tags, including the special START and END tags.
		/// </summary>
		public IReadOnlyBiGramSet<T> TagBiGrams
		{
			get; private set;
		}

		/// <summary>
		/// The open sequence of possible tags starting from <see cref="LinearChainCRF{I, T}.StartTag"/>, 
		/// as implied by <see cref="TagBiGrams"/>.
		/// </summary>
		public LazySequence<ISet<T>> ForwardTagsByIndex
		{
			get
			{
				return forwardTagsByIndex;
			}
		}

		/// <summary>
		/// The open sequence of possible tags ending to <see cref="LinearChainCRF{I, T}.EndTag"/>, 
		/// as implied by <see cref="TagBiGrams"/>.
		/// Indexing is reverse relative to the end, including <see cref="LinearChainCRF{I, T}.EndTag"/>.
		/// </summary>
		public LazySequence<ISet<T>> BackwardTagsByReverseIndex
		{
			get
			{
				return backwardTagsByReverseIndex;
			}
		}

		/// <summary>
		/// Cache holding the allowed tags for a given sequence length, 
		/// as implied by <see cref="TagBiGrams"/>.
		/// Default size is 256. Its content and settings are not serialized.
		/// </summary>
		public MRUCache<int, ISet<T>[]> AllowedTagsByIndexCache
		{
			get
			{
				return allowedTagsByIndexCache;
			}
		}

		#endregion

		#region Protected methods

		/// <summary>
		/// Specifies the evaluator implementation for the conditional random field over
		/// an input sequence.
		/// </summary>
		protected override SequenceEvaluator GetSequenceEvaluator(Vector weights, FeatureFunctionsProvider featureFunctionsProvider)
		{
			if (weights == null) throw new ArgumentNullException("weights");
			if (featureFunctionsProvider == null) throw new ArgumentNullException("featureFunctionsProvider");

			return new ConstrainedSequenceEvaluator(weights, this, featureFunctionsProvider);
		}

		#endregion

		#region Private methods

		private ISet<T> ComputeForwardTagsAtIndex(int index)
		{
			if (index < 0) throw new ArgumentException("index must not be negative.", "index");

			if (index == 0)
			{
				return new HashSet<T>(this.TagBiGrams.GetByFirst(this.StartTag).Select(biGram => biGram.Item2));
			}

			var previousTags = this.ForwardTagsByIndex[index - 1];

			var nextTags = from previousTag in previousTags
										 from nextBiGram in this.TagBiGrams.GetByFirst(previousTag)
										 select nextBiGram.Item2;

			var nextTagsSet = new HashSet<T>(nextTags);

			//var nextTagsSet = new HashSet<T>();

			//foreach (var previousTag in previousTags)
			//{
			//  foreach (var nextBiGram in this.TagBiGrams.GetByFirst(previousTag))
			//  {
			//    nextTagsSet.Add(nextBiGram.Item2);
			//  }
			//}

			return nextTagsSet;
		}

		private ISet<T> ComputeBackwardTagsAtReverseIndex(int reverseIndex)
		{
			if (reverseIndex < 0) throw new ArgumentException("reverseIndex must not be negative.", "reverseIndex");

			if (reverseIndex == 0)
			{
				var endSet = new HashSet<T>();
				
				endSet.Add(this.EndTag);
				
				return endSet;
			}

			var nextTags = this.BackwardTagsByReverseIndex[reverseIndex - 1];

			var previousTags = from nextTag in nextTags
												 from previousBiGram in this.TagBiGrams.GetBySecond(nextTag)
												 select previousBiGram.Item1;

			return new HashSet<T>(previousTags);
		}

		private ISet<T>[] ComputeAllowedTagsSequence(int sequenceLength)
		{
			if (sequenceLength < 0) throw new ArgumentException("sequenceLength must not be negative.", "sequenceLength");

			ISet<T>[] allowedTagsByIndex = new ISet<T>[sequenceLength + 1];

			for (int i = 0; i < sequenceLength; i++)
			{
				var forwardTags = this.ForwardTagsByIndex[i];
				var backwardTags = this.BackwardTagsByReverseIndex[sequenceLength - i];

				var allowedTags = new HashSet<T>();

				allowedTags.UnionWith(forwardTags);
				allowedTags.IntersectWith(backwardTags);
				allowedTags.Remove(this.StartTag);
				allowedTags.Remove(this.EndTag);

				allowedTagsByIndex[i] = allowedTags;
			}

			allowedTagsByIndex[sequenceLength] = this.BackwardTagsByReverseIndex[0];

			return allowedTagsByIndex;
		}

		private void InitializeTagSetSequences()
		{
			this.forwardTagsByIndex = new LazySequence<ISet<T>>(this.ComputeForwardTagsAtIndex);
			this.backwardTagsByReverseIndex = new LazySequence<ISet<T>>(this.ComputeBackwardTagsAtReverseIndex);
			this.allowedTagsByIndexCache = new MRUCache<int, ISet<T>[]>(this.ComputeAllowedTagsSequence, 256);
		}

		#endregion

		#region IDeserializationCallback Members

		public void OnDeserialization(object sender)
		{
			InitializeTagSetSequences();
		}

		#endregion
	}
}
