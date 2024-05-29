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



/*---------------------------------------------------------------------
-----------------Methods of MCFBlock not modified yet------------------
---------------------------------------------------------------------*/

void StochasticSolution::deserialize( const netCDF::NcGroup & group )
{
 netCDF::NcDim na = group.getDim( "NumArcs" );
 if( na.isNull() )
  v_x.clear();
 else {
  netCDF::NcVar fs = group.getVar( "FlowSolution" );
  if( fs.isNull() )
   v_x.clear();
  else {
   v_x.resize( na.getSize() );
   fs.getVar( v_x.data() );
   }
  }

 netCDF::NcDim nn = group.getDim( "NumNodes" );
 if( nn.isNull() )
  v_pi.clear();
 else {
  netCDF::NcVar ps = group.getVar( "Potentials" );
  if( ps.isNull() )
   v_pi.clear();
  else {
   v_pi.resize( nn.getSize() );
   ps.getVar( v_pi.data() );
   }
  }
 }  // end( MCFSolution::deserialize )

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

void StochasticSolution::serialize( netCDF::NcGroup & group ) const
{
 // always call the method of the base class first
 Solution::serialize( group );

 std::vector< size_t > startp = { 0 };

 if( ! v_x.empty() ) {
  netCDF::NcDim na = group.addDim( "NumArcs" , v_x.size() );

  std::vector< size_t > countpa = { v_x.size() };

  ( group.addVar( "FlowSolution" , netCDF::NcDouble() , na ) ).putVar(
					      startp , countpa , v_x.data() );
  }

 if( v_pi.empty() )
  return;

 netCDF::NcDim nn = group.addDim( "NumNodes" ,  v_pi.size() );
 std::vector< size_t > countpn = { v_pi.size() };
 ( group.addVar( "Potentials" , netCDF::NcDouble() , nn ) ).putVar(
					     startp , countpn , v_pi.data() );
 
 }  // end( MCFSolution::serialize )