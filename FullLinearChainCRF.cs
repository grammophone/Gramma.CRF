using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Gramma.Vectors;

namespace Gramma.CRF
{
	/// <summary>
	/// Linear Chain Conditional Random Field. Must be subclassed for use.
	/// All possible tag combinations of a given tag set are computed.
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
	public abstract class FullLinearChainCRF<I, T> : LinearChainCRF<I, T>
	{
		#region Auxilliary classes

		/// <summary>
		/// Represents the evaluation tool of a condition random field over an input sequence.
		/// </summary>
		private class FullSequenceEvaluator : SequenceEvaluator
		{
			#region Private fields

			/// <summary>
			/// The array of tags, NOT including the START and END tags.
			/// </summary>
			private T[] tagSet;

			private double[][,] g;

			private double[][,] expG;

			private double[,] u;

			private double[,] forwardVector;

			private double[,] backwardVector;

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

			public FullSequenceEvaluator(Vector w, FullLinearChainCRF<I, T> crf, FeatureFunctionsProvider featureFunctionsProvider)
				: base(w, crf, featureFunctionsProvider)
			{
				tagSet = crf.TagSet;
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
			public double[][,] G
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
			public double[][,] ExpG
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
			public double[,] U
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
			public double[,] ForwardVector
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
			public double[,] BackwardVector
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
				int tagSetCount = tagSet.Length;

				int sequenceCount = this.OutputLength;

				T[] y = new T[sequenceCount];

				var u = this.U;

				double max;

				var g = this.G;

				// WARNING: Our base case is different from most tutorials which specify:
				// yn = max_v u[k, v]
				// We use the following for consistency instead, because, throughout our whole approach,
				// our feature functions span up to the END position, including.
				// yn = max_v u[k, v] + gn[v, 0];
				// So, We can go ahead directly to the recursion which has the same form as above.

				// Recursion.

				int ykm1;

				int yk = 0;

				for (int km1 = y.Length - 1; km1 >= 0; km1--)
				{
					var gk = g[km1 + 1];

					max = double.NegativeInfinity;

					ykm1 = 0;

					for (int v = 0; v < tagSetCount; v++)
					{
						double candidate = u[km1, v] + gk[v, yk];

						if (candidate > max)
						{
							max = candidate;
							ykm1 = km1;
						}
					}

					y[km1] = tagSet[ykm1];

					yk = ykm1;
				}

				return y;
			}

			protected override IVector ComputeFeatureFunctionsExpectations()
			{
				var α = this.ForwardVector;
				var β = this.BackwardVector;
				var expG = this.ExpG;

				Vector E = new Vector(this.w);

				// i = 0
				{
					var expG0 = expG[0];

					for (int y0 = 0; y0 < tagSet.Length; y0++)
					{
						var factor = expG0[0, y0] * β[y0, 0];

						E.AddInPlace((f(startTag, tagSet[y0], x, 0).Add(h(tagSet[y0], x, 0))).Scale(factor));
					}
				}

				// 0 < i < n
				for (int i = 1; i < this.OutputLength; i++)
				{
					var expGi = expG[i];

					for (int yi = 0; yi < tagSet.Length; yi++)
					{
						var β_yi_i = β[yi, i];

						double totalFactor = 0.0;

						for (int yim1 = 0; yim1 < tagSet.Length; yim1++)
						{
							double factor = α[i - 1, yim1] * expGi[yim1, yi] * β_yi_i;

							totalFactor += factor;

							E.AddInPlace(f(tagSet[yim1], tagSet[yi], x, i).Scale(factor));
						}

						E.AddInPlace(h(tagSet[yi], x, i).Scale(totalFactor));
					}
				}

				// i = n
				{
					int n = this.OutputLength;
					int nm1 = this.OutputLength - 1;

					var expGn = expG[n];

					double totalFactor = 0.0;

					for (int ynm1 = 0; ynm1 < tagSet.Length; ynm1++)
					{
						double factor = α[nm1, ynm1] * expGn[ynm1, 0];

						totalFactor += factor;

						E.AddInPlace(f(tagSet[ynm1], endTag, x, n).Scale(factor));
					}

					E.AddInPlace(h(endTag, x, n).Scale(totalFactor));
				}

				return E.DivideInPlace(this.Z);
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

				double zu = 0.0;

				for (int i = 0; i < tagSet.Length; i++)
				{
					zu += β[0, i] * expG0[0, i];
				}

				return zu / c0;
			}

			#endregion

			#region Private methods

			private double[][,] ComputeG()
			{
				if (this.g != null) return this.g;

				int sequenceCount = this.OutputLength;

				double[][,] g = new double[sequenceCount + 1][,];

				// Compute special start case i = 0.
				{
					var gi = new double[1, tagSet.Length];

					g[0] = gi;

					for (int k = 0; k < tagSet.Length; k++)
					{
						T y0 = tagSet[k];

						gi[0, k] = w * (f(startTag, y0, x, 0).Add(h(y0, x, 0)));
					}
				}

				// Compute for mid part of sequence x.
				for (int i = 1; i < sequenceCount; i++)
				{
					var gi = new double[tagSet.Length, tagSet.Length];
					g[i] = gi;

					for (int l = 0; l < tagSet.Length; l++)
					{
						T yi = tagSet[l];

						double unigramComponent = w * h(yi, x, i);

						for (int k = 0; k < tagSet.Length; k++)
						{
							T yim1 = tagSet[k];

							gi[k, l] = w * f(yim1, yi, x, i) + unigramComponent;
						}
					}

				}

				// Compute special end case i = |x|.
				{
					var gi = new double[tagSet.Length, 1];

					g[sequenceCount] = gi;

					double unigramComponent = w * h(endTag, x, sequenceCount);

					for (int k = 0; k < tagSet.Length; k++)
					{
						T ynm1 = tagSet[k];

						gi[k, 0] = w * f(ynm1, endTag, x, sequenceCount) + unigramComponent;
					}
				}

				return g;
			}

			private double[][,] ComputeExpG()
			{
				var g = this.G;

				double[][,] expG = new double[g.Length][,];

				for (int i = 0; i < g.Length; i++)
				{
					var gi = g[i];

					var K = gi.GetLength(0);
					var L = gi.GetLength(1);

					double[,] expGi = new double[K, L];

					expG[i] = expGi;

					for (int k = 0; k < K; k++)
					{
						for (int l = 0; l < L; l++)
						{
							expGi[k, l] = Math.Exp(gi[k, l]);
						}
					}
				}

				return expG;
			}

			private double[,] ComputeU()
			{
				int tagSetCount = tagSet.Length;

				int sequenceCount = this.OutputLength;

				var u = new double[sequenceCount + 1, tagSetCount];

				var g = this.G;

				// Base case k = 0
				{
					var g0 = g[0];

					for (int v = 0; v < tagSetCount; v++)
					{
						u[0, v] = g0[0, v];
					}
				}

				// Recursion
				{
					for (int k = 1; k < sequenceCount; k++)
					{
						var gk = g[k];

						for (int v = 0; v < tagSetCount; v++)
						{
							double max = Double.NegativeInfinity;

							for (int m = 0; m < tagSetCount; m++)
							{
								double candidate = u[k - 1, m] + gk[m, v];

								if (candidate > max) max = candidate;
							}

							u[k, v] = max;
						}
					}
				}

				// k = n.
				{
					double max = Double.NegativeInfinity;

					var gn = g[sequenceCount];

					for (int q = 0; q < tagSetCount; q++)
					{
						double candidate = u[sequenceCount - 1, q] + gn[q, 0];

						if (candidate > max) max = candidate;
					}

					u[sequenceCount, 0] = max;
				}

				return u;
			}

			private double[,] ComputeForwardVector()
			{
				int tagSetCount = tagSet.Length;

				int sequenceCount = this.OutputLength;

				var α = new double[sequenceCount, tagSetCount];

				this.c = new double[sequenceCount];

				var expG = this.ExpG;

				// This will hold the the normalizer for each level.
				double ck;
				// This will hold the inverse of the normalizer for each level.
				double invck;

				// Base case.

				invck = 0.0;

				var expG0 = expG[0];

				for (int v = 0; v < tagSetCount; v++)
				{
					double unnormalized = expG0[0, v];
					α[0, v] = unnormalized;
					invck += unnormalized;
				}

				ck = 1 / invck;
				c[0] = ck;

				// Rescale.

				for (int v = 0; v < α.GetLength(1); v++)
				{
					α[0, v] *= ck;
				}

				// Recursion.

				for (int k = 1; k < sequenceCount; k++)
				{
					int km1 = k - 1;

					var expGk = expG[k];

					invck = 0.0;

					for (int v = 0; v < tagSetCount; v++)
					{
						double sum = 0.0;

						for (int u = 0; u < tagSetCount; u++)
						{
							sum += α[km1, u] * expGk[u, v];
						}

						α[k, v] = sum;

						invck += sum;
					}

					ck = 1 / invck;
					c[k] = ck;

					// Rescale.

					for (int v = 0; v < α.GetLength(1); v++)
					{
						α[k, v] *= ck;
					}
				}

				return α;
			}

			private double[,] ComputeBackwardVector()
			{
				// Force computing the forward vector first, if not already computed,
				// in order to create the c[k] normalization weights.
				var forwardVector = this.ForwardVector;

				int tagSetCount = tagSet.Length;

				int sequenceCount = this.OutputLength;

				var β = new double[tagSetCount, sequenceCount];

				var expG = this.ExpG;

				double ck;

				// Base case.

				var expGn = expG[sequenceCount];

				ck = c[sequenceCount - 1];

				for (int u = 0; u < tagSetCount; u++)
				{
					β[u, sequenceCount - 1] = expGn[u, 0] * ck;
				}

				// Recursion.

				for (int k = sequenceCount - 2; k >= 0; k--)
				{
					int kp1 = k + 1;

					var expGkp1 = expG[kp1];

					ck = c[k];

					for (int u = 0; u < tagSetCount; u++)
					{
						double sum = 0.0;

						for (int v = 0; v < tagSetCount; v++)
						{
							sum += expGkp1[u, v] * β[v, kp1];
						}

						β[u, k] = sum * ck;
					}
				}

				return β;
			}

			#endregion
		}

		#endregion

		#region Construction

		/// <summary>
		/// Create.
		/// </summary>
		/// <param name="tagSet">All the tags, excluding START and END.</param>
		/// <param name="startTag">The special tag used to mark the start of a sentence, implied at index -1. Does not belong to the normal tags.</param>
		/// <param name="endTag">The special tag used to mark the end of a sentence, implied at index |x|. Does not belong to the normal tags.</param>
		/// <param name="featureFunctionsProviderFactory">The provider of feature functions.</param>
		public FullLinearChainCRF(T[] tagSet, T startTag, T endTag, FeatureFunctionsProviderFactory featureFunctionsProviderFactory)
			: base(startTag, endTag, featureFunctionsProviderFactory)
		{
			if (tagSet == null) throw new ArgumentNullException("tagSet");

			this.TagSet = tagSet;
		}

		#endregion

		#region Public properties

		/// <summary>
		/// All the tags, excluding START and END.
		/// </summary>
		public T[] TagSet { get; private set; }

		#endregion

		#region Protected methods

		/// <summary>
		/// Specifies the evaluator implementation for the conditional random field over
		/// an input sequence.
		/// </summary>
		protected override SequenceEvaluator GetSequenceEvaluator(Vector w, FeatureFunctionsProvider featureFunctionsProvider)
		{
			if (w == null) throw new ArgumentNullException("w");
			if (featureFunctionsProvider == null) throw new ArgumentNullException("featureFunctionsProvider");

			return new FullSequenceEvaluator(w, this, featureFunctionsProvider);
		}

		#endregion
	}
}
