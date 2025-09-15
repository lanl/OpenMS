#pragma once

#if !defined(index_type)
#  cmakedefine signed_index_type 1
#  if defined(signed_index_type)
#    define index_type std::int64_t
#  else
#    define index_type std::uint64_t
#  endif
#endif

using Index = index_type;

