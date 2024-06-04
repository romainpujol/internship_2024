/*--------------------------------------------------------------------------*/
/*----------------------- METHODS OF StochasticSolution---------------------*/
/*--------------------------------------------------------------------------*/

void StochasticSolution::sum( const Solution * solution , double multiplier )
{
 auto MCFS = dynamic_cast< const StochasticSolution * >( solution );
 if( ! MCFS )
  throw( std::invalid_argument( "solution is not a StochasticSolution" ) );

 if( ! atoms.empty() ) {
  if( atoms.size() != MCFS->atoms.size() )
   throw( std::invalid_argument( "incompatible size" ) );

  for( int i = 0 ; i < atoms.size() ; ++i ){
    if (atoms[i].size() != (MCFS->atoms)[i].size()){
        throw( std::invalid_argument( "incompatible dimension" ) );
    }
    else{
        for (int j=0 ; j<atoms[i].size(); ++j)
            atoms[i][ j ] += MCFS->atoms[ i ][j]* multiplier;
    }
  }
}
    // summing probabilities would make no sense at all.
}  // end( MCFSolution::sum )

StochasticSolution * StochasticSolution::clone( bool empty ) const
{
 auto *sol = new StochasticSolution();

 if( empty ) {
  if( ! p.empty() )
   sol->p.resize( p.size() );

  if( ! atoms.empty() )
   sol->atoms.resize( atoms.size() );
  }
 else {
  sol->atoms = atoms;
  sol->p = p;
  }

 return( sol );

 }  // end( StochasticSolution::clone )

StochasticSolution * StochasticSolution::scale( double factor ) const
{
 auto * sol = MCFSolution::clone( true );
 int m = atoms.size();

 if( m > 0 )
  for( int i = 0 ; i < m ; ++i ){
   for (int j = 0; i<atoms[i].size();j++){
    sol->atoms[i][j]*=factor;
   }
  } 

 // scaling probability makes no sense at all. 

 return( sol );

 }  // end( StochasticSolution::scale )

void StochasticSolution::deserialize( const netCDF::NcGroup & group )
{
 netCDF::NcDim n = group.getDim( "n" ); // n is the size of the distribution
 netCDF::NcDim d= group.getDim( "d"); // dimension of the space

 if( n.isNull() || d.isNull())
  p.clear();
  atoms.clear();
 else {
  netCDF::NcVar as = group.getVar( "AtomSolution" );
  dim=d
  if( as.isNull() ) 
   atoms.clear();
  else {
   p.resize( n.getSize() );
   as.getVar( atoms.data() );
   }
  }
 }  // end( MCFSolution::deserialize )

void StochasticSolution::serialize( netCDF::NcGroup & group ) const
{
 if( ! p.empty() && ! atoms.empty()) {

  // always call the method of the base class first
  Solution::serialize( group );

  int d = atoms.size() / p.size(); 

  netCDF::NcDim n = group.addDim( "n" , p.size() ); 

  ( group.addVar( "as" , netCDF::NcDouble() , n*d ) ).putVar( atoms.data() );
  ( group.addVar( "p" , netCDF::NcDouble() , n ) ).putVar( p.data() );

  }
 }



/*---------------------------------------------------------------------
-----------------Methods of MCFBlock not modified yet------------------
---------------------------------------------------------------------*/


/*--------------------------------------------------------------------------*/

void MCFSolution::read( const Block * block )
{
 auto MCFB = dynamic_cast< const MCFBlock * >( block );
 if( ! MCFB )
  throw( std::invalid_argument( "block is not a MCFBlock" ) );

 // read flows- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 if( ! v_x.empty() ) {
  if( v_x.size() < MCFB->get_NArcs() )
   v_x.resize( MCFB->get_NArcs() );

  MCFB->get_x( v_x.begin() );
  }

 // read potentials - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 if( MCFB->E.empty() && MCFB->dE.empty() )  // no potentials available
  return;

 if( ! v_pi.empty() ) {
  if( v_pi.size() < MCFB->get_NNodes() )
   v_pi.resize( MCFB->get_NNodes() );

  MCFB->get_pi( v_pi.begin() );
  }
 }  // end( MCFSolution::read )

/*--------------------------------------------------------------------------*/

void MCFSolution::write( Block * block ) 
{
 auto MCFB = dynamic_cast< MCFBlock * >( block );
 if( ! MCFB )
  throw( std::invalid_argument( "block is not a MCFBlock" ) );

 // write flows - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 if( ! v_x.empty() ) {
  if( v_x.size() < MCFB->get_NStaticArcs() )
   throw( std::invalid_argument( "incompatible flow size" ) );

  MCFB->set_x( v_x.begin() );
  }

 // write potentials- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 if( v_pi.empty() )  // no potentials to write
  return;

 if( MCFB->E.empty() && MCFB->dE.empty() )  // no Constraint to write to
  return;

 if( v_pi.size() < MCFB->get_NStaticNodes() )
  throw( std::invalid_argument( "incompatible potential size" ) );

 MCFB->set_pi( v_pi.begin() );

 // write reduced costs (if any)- - - - - - - - - - - - - - - - - - - - - - -
  
 if( MCFB->UB.empty() && MCFB->dUB.empty() )  // no bounds to write to
  return;

 MCFBlock::Index i = 0;

 // static part
 for( auto ubi = MCFB->UB.begin() ; ubi != MCFB->UB.end() ; ++i )
  (ubi++)->set_dual( MCFB->get_C( i ) + v_pi[ MCFB->SN[ i ] - 1 ]
		                      - v_pi[ MCFB->EN[ i ] - 1 ] );
 // dynamic part
 for( auto dubi = MCFB->dUB.begin() ;
      ( dubi != MCFB->dUB.end() ) && ( i < MCFB->get_NArcs() ) ; ++i )
  (dubi++)->set_dual( MCFB->get_C( i ) + v_pi[ MCFB->SN[ i ] - 1 ]
		                       - v_pi[ MCFB->EN[ i ] - 1 ] );
 }  // end( MCFSolution::write )

/*--------------------------------------------------------------------------*/

