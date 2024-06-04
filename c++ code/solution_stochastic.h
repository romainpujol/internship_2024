#include <stdio.h>

class StochasticSolution{

    public:
    void deserialize( const netCDF::NcGroup & group );
    void read( const Block * const block );
    void write( Block * const block );
    StochasticSolution * scale( double factor );
    void sum( const Solution * solution , double multiplier );
    StochasticSolution * clone( bool empty = false ) const;

    private:
    std::vector <double> p; // probabilities p[i] corresponds to the probability of observing atoms[i]
    std::vector <double> atoms; //atoms
    int dim; // dimension
    
    }

/*--------------------------------------------------------------------------*/
 /// de-serialize a :Solution out of netCDF::NcGroup
 /** The method takes a netCDF::NcGroup supposedly containing all the
  * information required to de-serialize the :Solution, and produces a "full"
  * Solution object as a result. Most likely, the netCDF::NcGroup has been
  * produced by calling serialize() with a previously existing :Solution (of
  * the very same type as this one), but individual :Solution should openly
  * declare the format of their :Solution so that possibly a netCDF::NcGroup
  * containing some pre-computed solution can be constructed from scratch
  * whenever this is useful.
  *
  * This method is pure virtual, as it clearly has to be implemented by
  * derived classes. */

//void deserialize( const netCDF::NcGroup & group ){}

/** @name Methods describing the behavior of a Solution
 *  @{ */

 /// read the solution from the given Block
 /** This method reads the solution of the given Block and stores it in this
  * Solution. A Solution object can be "configured" to take only a specific
  * part of the Block solution status: it is an error if block does not have
  * the required part (and, a fortiori, if block is not the right Block). */

// virtual void read( const Block * const block ) = 0;

/*--------------------------------------------------------------------------*/
 /// write the solution in the given Block
 /** This method writes the solution currently stored in this Solution in the
  * given Block. A Solution object can be "configured" to take only a specific
  * part of the Block solution status: it is an error if block does not have
  * the required part (and, a fortiori, if block is not the right Block). */

// virtual void write( Block * const block ) = 0;

/*--------------------------------------------------------------------------*/
 /// serialize a :Solution into a netCDF::NcGroup
 /** The method takes a (supposedly, "full") Solution object and serializes
  * it into the provided netCDF::NcGroup, so that it can possibly be read by
  * deserialize() (of a :Solution of the very same type as this one).
  *
  * The method of the base class just creates and fills the "type" attribute
  * (with the right name, thanks to the classname() method) and the optional
  * "name" attribute. Yet
  *
  *     serialize() OF ANY :Solution SHOULD CALL Solution::serialize()
  *
  * While this currently does so little that one might well be tempted to
  * skip the call and just copy the three lines of code, enforcing this
  * standard is forward-looking since in this way any future revision of the
  * base Solution class may add other mandatory/optional fields: as soon as
  * they are managed by the (revised) method of the base class, they would
  * then be automatically dealt with by the derived classes without them even
  * knowing it happened. */

 //virtual void serialize( netCDF::NcGroup & group ) const {group.putAtt( "type" , classname() );}


 /// returns a scaled version of this Solution
 /** This method constructs and returns a scaled version of this Solution,
  * where each of the solution information is scaled by the given double
  * value. A Solution object can be "configured" to take only a specific
  * part of the Block solution status: the scaled version of a Solution
  * object obviously "shares the same configuration" as the original object.
  *
  * Scaling by a double is a "somewhat dangerous" operation: it is quite
  * natural if all solution information is "double", which is what is most
  * likely to happen most of the time, but not in all cases. For instance,
  * some Block may correspond to combinatorial problems whose solutions are
  * essentially combinatorial objects (paths/cuts on a graph ...), for which
  * "scaling" makes little sense. Yet, these problems are typically also
  * represented in terms of subspaces of \R^n, and therefore one might
  * expect scaling to be possible. However, it is clear that scaling may
  * destroy some of the properties that solutions have: for instance, a
  * path in a graph can be represented by means of a predecessor function,
  * but a scaled path can not -- in the sense that it needs at least another
  * information, the scaling factor, or to be transformed into a different
  * format, such as the amount of "flow" going on each arc of the graph.
  *
  * This implies that there might be different Solution objects relative to
  * a given Block; say, the "original ones" corresponding to combinatorial
  * solutions (a path, represented by a predecessor function) and the "scaled
  * ones) produced by scaling and/or summing (see sum()) below (say, a double
  * for each arc of the graph). Thus, scale() might return a Solution object
  * that, while being appropriate for the original Block, may in fact be "of
  * a different type" than the originating one". The requirement is that the
  * newly created Solution must be a "general" one, in the sense that it
  * makes sense (obviously) to scale it, and also to *sum* it with other
  * Solution objects, see sum(). */

 //virtual Solution * scale( double factor ) const = 0;

/*--------------------------------------------------------------------------*/
 /// adds a scaled version of the given Solution to this Solution
 /** This method adds a scaled version of the given Solution (see scale()) to
  * this Solution. A Solution object can be "configured" to take only a
  * specific part of the Block solution status: this means that even if the
  * Solution object pointed by solution has "more information" than the
  * current one, only the relevant part will be extracted and summed to that
  * of the current one, so that "the original configuration is preserved"
  * even after this operation. It is an error if solution does not have the
  * required part (and, a fortiori, if it is not the right Solution),
  * resulting in an exception being thrown.
  *
  * As discussed in scale(), scaling by a double is a "somewhat dangerous"
  * operation that may not necessarily make sense for all kinds of solution
  * information, in particular discrete ones. The same potentially holds for
  * sums (many combinatorial structures are closed under sum but not all are),
  * and a fortiori for "sum with a scaled object". Thus, some Solution may not
  * be able to properly implement this operation without fundamentally alter
  * their own internal representation, which is not supposed to happen. Thus,
  * 
  *    IT IS NOT NECESSARILY SAFE TO CALL sum() ON A Solution JUST
  *    PRODUCED BY Block::get_Solution()
  *
  * although in general it should always be possible to "configure the
  * Solution", by using the corresponding Configuration object to instruct
  * the Block to produce the kind of Solution object for which it is safe.
  * Furthermore,
  *
  *    IT IS SAFE TO CALL sum() ON A SOLUTION CONSTRUCTED BY scale()
  *
  * That is, scale() has to report a "general" Solution, one for which it
  * makes sense *both* scale it and sum it with other Solution objects.
  * Note that the idea is that is must be always possible to use "less
  * general" solution objects as the solution parameter in sum(): the
  * recipient (current) Solution object must be "general" for sum() to be
  * possible, but the summed one need not be. Of course, all this must be
  * entirely handled by the (different variants of) Solution.
  *
  * It should be remarked that there could be "intermediate" types of
  * Solution objects between the "less general" and the "more general" ones.
  * For instance, some discrete structures are closed under sum, or even
  * scaled sum where the scalar has appropriate properties (say, it's an
  * integer). Thus, it may not be efficient to require scale() to return the
  * "more general" Solution. Yet, handling these special cases should always
  * be possible by requiring the Block to produce "the right kind of Solution
  * object" by means of its Configuration. */

 //virtual void sum( const Solution * solution , double multiplier ) = 0;

/*--------------------------------------------------------------------------*/
/// returns a clone of this Solution
/** This method creates and returns a Solution of the same type of this
 * Solution. If the parameter empty is true, then the returned Solution is
 * "empty", i.e., the solution information is not passed over, otherwise the
 * new Solution is a complete copy of the current one. A Solution object can
 * be "configured" to take only a specific part of the Block solution status:
 * the cloned object obviously "shares the same configuration" as the
 * original object.
 *
 * Note that clone() and scale( 1 ) return in principle the same Solution.
 * However, scale( 1 ) must return a "general solution" (see comments in
 * scale() and sum()), whereas clone() can return exactly the same type of
 * solution as the current one, i.e., a "less general" one if this is. */

//virtual Solution * clone( bool empty = false ) const = 0;

/*--------------------------------------------------------------------------*/
/*-------------------------- PROTECTED METHODS -----------------------------*/
/*--------------------------------------------------------------------------*/
/** @name Protected methods for printing and serializing
    @{ */

 /// print information about the Solution on an ostream
 /** Protected method intended to print information about the Solution; it is
  * virtual so that derived classes can print their specific information in
  * the format they choose. */

 //virtual void print( std::ostream &output ) const {output << "Solution [" << this << "]";}
