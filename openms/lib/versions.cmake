# for each dependency track both current and previous id (the variable for the latter must contain PREVIOUS)
# to be able to auto-update them

# Apr 24, The versions numbers (tags) to be updated (TODO)

set(OMS_TRACKED_CMAKEKIT_TAG 3cbfe7c1e2e2667964b737e6abcc44d173fb9775)

#=========================================================================
# Boost explicitly downgraded to 1.59 from 1.68
#=========================================================================
set(OMS_TRACKED_BOOST_VERSION 1.59)
set(OMS_TRACKED_BOOST_PREVIOUS_VERSION 1.68)
set(OMS_INSTALL_BOOST_VERSION 1.70.0)
set(OMS_INSTALL_BOOST_PREVIOUS_VERSION 1.70.0)
set(OMS_INSTALL_BOOST_URL_HASH 882b48708d211a5f48e60b0124cf5863c1534cd544ecd0664bb534a4b5d506e9)
set(OMS_INSTALL_BOOST_PREVIOUS_URL_HASH 882b48708d211a5f48e60b0124cf5863c1534cd544ecd0664bb534a4b5d506e9)

#=========================================================================
## MEEP (again, version, tags to be updated)
#=========================================================================
#set(OMS_TRACKED_MEEP_VERSION 1.26.0)
#set(OMS_TRACKED_MEEP_PREVIOUS_VERSION 1.26.0)
#set(OMS_INSTALL_MEEP_VERSION 1.26.0)
#set(OMS_INSTALL_MEEP_PREVIOUS_VERSION 1.26.0)
#set(OMS_INSTALL_MEEP_URL_HASH SHA256=xx)
#set(OMS_INSTALL_MEEP_PREVIOUS_URL_HASH MD5=xx)

#=========================================================================
# (TA's dependencies versions are managed within TA itself). 
# But, we need to check the consistency with other external libs (TODO)
#### may need to update INSTALL.md manually with the CUDA-specific version
# again, versions and tags to be updated
#=========================================================================
#set(OMS_TRACKED_TA_VERSION 1.0.0)
#set(OMS_TRACKED_TA_PREVIOUS_VERSION 1.0.0)
#set(OMS_INSTALL_TA_VERSION 1.0.0)
#set(OMS_INSTALL_TA_PREVIOUS_VERSION 1.0.0)
#set(OMS_INSTALL_TA_URL_HASH SHA256=xxx)
#set(OMS_INSTALL_TA_PREVIOUS_URL_HASH MD5=xxx)


