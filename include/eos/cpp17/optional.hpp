/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/cpp17/optional.hpp
 *
 * Copyright 2018 Patrik Huber
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#ifndef EOS_OPTIONAL_HPP_
#define EOS_OPTIONAL_HPP_

#ifdef __APPLE__
  #include "eos/cpp17/detail/akrzemi1_optional.hpp"
  namespace eos {
    //namespace cpp17 = ::akrzemi1;
      //namespace cpp17 = ::asdf::akrzemi1;
      namespace cpp17 = ::std::experimental;
  }
#else
  #include <optional>
  namespace eos {
    namespace cpp17 = ::std;
  }
#endif

#endif /* EOS_OPTIONAL_HPP_ */
