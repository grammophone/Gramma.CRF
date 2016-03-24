using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Gramma.CRF
{
	/// <summary>
	/// A lazy infinite sequence of items of type <typeparamref name="T"/>.
	/// </summary>
	/// <typeparam name="T">The type of items in the sequence.</typeparam>
	/// <remarks>
	/// The sequence is implied by a user-supplied item function.
	/// When an item of the sequence is queried, it is guaranteed that all previous items
	/// have been computed first. This makes possible recursive computations.
	/// </remarks>
	[Serializable]
	public class LazySequence<T>
	{
		#region Private fields

		private Func<int, T> itemFunction;

		private List<T> items;

		#endregion

		#region Construction

		/// <summary>
		/// Create.
		/// </summary>
		/// <param name="itemFunction">The function defining each item. It can depend on previous items, not on next items.</param>
		/// <remarks>
		/// When an item of the sequence is queried, it is guaranteed that all previous items
		/// have been computed first. This makes possible recursive computations.
		/// If <see cref="LazySequence{T}"/> is used in serialization, the <paramref name="itemFunction"/>
		/// should comply to the serialization requirements.
		/// </remarks>
		public LazySequence(Func<int, T> itemFunction)
		{
			if (itemFunction == null) throw new ArgumentNullException("itemFunction");

			this.itemFunction = itemFunction;
			this.items = new List<T>();
		}

		#endregion

		#region Public properties

		/// <summary>
		/// Get an item of the sequence, lazily computing all missing items up to the given index if not
		/// already done so.
		/// </summary>
		/// <param name="index">The index of the item.</param>
		/// <returns>Returns the item at the given index.</returns>
		/// <remarks>
		/// When an item of the sequence is queried, it is guaranteed that all previous items
		/// have been computed first. This makes possible recursive computations.
		/// </remarks>
		public T this[int index]
		{
			get
			{
				lock (items)
				{
					for (int i = items.Count; i <= index; i++)
					{
						items.Add(itemFunction(i));
					}
				}

				return items[index];
			}
		}

		#endregion
	}
}
