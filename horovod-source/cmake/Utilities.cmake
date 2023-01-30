# Parse framework version
function(PARSE_VERSION VERSION OUTPUT)
    string(FIND "${VERSION}" "dev" RESULT)
    if (NOT RESULT EQUAL "-1")
        set(${OUTPUT} "9999999999" PARENT_SCOPE)
    else()
        string(REGEX MATCH "^[0-9]+(\\.[0-9]+)?(\\.[0-9]+)?(\\.[0-9]+)?" VERS_MATCHED "${VERSION}")
        if (VERS_MATCHED)
            string(REPLACE "." ";" VERS_MATCHED "${VERS_MATCHED}")
            set(MULT "1000000000")
            set(ACC "0")
            foreach(_NUM IN LISTS VERS_MATCHED)
                math(EXPR ACC "${ACC} + ${_NUM} * ${MULT}")
                math(EXPR MULT "${MULT} / 1000")
            endforeach()
            set(${OUTPUT} ${ACC} PARENT_SCOPE)
        endif()
    endif()
endfunction()

# Set output dir for library
macro(SET_OUTPUT_DIR)
    string(REGEX REPLACE "${PROJECT_SOURCE_DIR}(/)?" "" SUB_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach( _VAR in ITEMS CMAKE_LIBRARY_OUTPUT_DIRECTORY CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO)
        if(DEFINED ${_VAR})
            set(${_VAR} "${${_VAR}}/${SUB_FOLDER}")
        endif()
    endforeach()
endmacro()

# Set build architecture flags
macro(SET_BUILD_ARCH_FLAGS FLAGS)
    set(HOROVOD_BUILD_ARCH_FLAGS $ENV{HOROVOD_BUILD_ARCH_FLAGS})
    if(NOT DEFINED HOROVOD_BUILD_ARCH_FLAGS)
        execute_process(COMMAND bash "-c" "gcc -march=native -E -v - </dev/null 2>&1 | grep cc1"
                        OUTPUT_VARIABLE COMPILER_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        string(REPLACE " " ";" COMPILER_FLAGS "${COMPILER_FLAGS}")
        foreach(_ARG ${FLAGS})
            string(REPLACE "-m" "" _SUB_ARG ${_ARG})
            if(";${COMPILER_FLAGS};" MATCHES ";-m${_SUB_ARG};" OR ";${COMPILER_FLAGS};" MATCHES ";+${_SUB_ARG};")
                list(APPEND HOROVOD_BUILD_ARCH_FLAGS ${_ARG})
            endif()
        endforeach()
        string(REPLACE ";" " " HOROVOD_BUILD_ARCH_FLAGS "${HOROVOD_BUILD_ARCH_FLAGS}")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HOROVOD_BUILD_ARCH_FLAGS}")
    message(STATUS "Build architecture flags: ${HOROVOD_BUILD_ARCH_FLAGS}")
endmacro()

# Set GPU OP
macro(SET_GPU_OP PARAM SUPPORTED)
    set(${PARAM} $ENV{${PARAM}})
    string(REPLACE ";" "|" REG "${SUPPORTED}")
    if(DEFINED HOROVOD_GPU_OPERATIONS)
        if(DEFINED ${PARAM})
            message(FATAL_ERROR "Cannot specify both HOROVOD_GPU_OPERATIONS and ${PARAM} options. "
                                "Try unsetting one of these variables and reinstalling.")
        endif()
        set(${PARAM} "${HOROVOD_GPU_OPERATIONS}")
    elseif(DEFINED ${PARAM} AND NOT "${${PARAM}}" MATCHES "^(${REG})$")
        string(REPLACE ";" ", " SUP "${SUPPORTED}")
        message(FATAL_ERROR "${PARAM}=${${PARAM}} is invalid, supported values are ${SUP}.")
    endif()
endmacro()

# Convert char to ASCII decimal
function(CONVERT_TO_ASCII_DEC NUMBER CHAR)
    set(ASCII_A "65")
    set(ASCII_B "66")
    set(ASCII_C "67")
    set(ASCII_D "68")
    set(ASCII_E "69")
    set(ASCII_F "70")
    set(ASCII_G "71")
    set(ASCII_H "72")
    set(ASCII_I "73")
    set(ASCII_J "74")
    set(ASCII_K "75")
    set(ASCII_L "76")
    set(ASCII_M "77")
    set(ASCII_N "78")
    set(ASCII_O "79")
    set(ASCII_P "80")
    set(ASCII_Q "81")
    set(ASCII_R "82")
    set(ASCII_S "83")
    set(ASCII_T "84")
    set(ASCII_U "85")
    set(ASCII_V "86")
    set(ASCII_W "87")
    set(ASCII_X "88")
    set(ASCII_Y "89")
    set(ASCII_Z "90")
    set(ASCII_a "97")
    set(ASCII_b "98")
    set(ASCII_c "99")
    set(ASCII_d "100")
    set(ASCII_e "101")
    set(ASCII_f "102")
    set(ASCII_g "103")
    set(ASCII_h "104")
    set(ASCII_i "105")
    set(ASCII_j "106")
    set(ASCII_k "107")
    set(ASCII_l "108")
    set(ASCII_m "109")
    set(ASCII_n "110")
    set(ASCII_o "111")
    set(ASCII_p "112")
    set(ASCII_q "113")
    set(ASCII_r "114")
    set(ASCII_s "115")
    set(ASCII_t "116")
    set(ASCII_u "117")
    set(ASCII_v "118")
    set(ASCII_w "119")
    set(ASCII_x "120")
    set(ASCII_y "121")
    set(ASCII_z "122")
    if(DEFINED ASCII_${CHAR})
        set(${NUMBER} "${ASCII_${CHAR}}" PARENT_SCOPE)
    else()
        set(${NUMBER} "${CHAR}" PARENT_SCOPE)
    endif()
endfunction()
