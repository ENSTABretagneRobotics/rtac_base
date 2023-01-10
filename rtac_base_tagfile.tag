<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile>
  <compound kind="file">
    <name>geometry.h</name>
    <path>/home/pnarvor/work/narval/code/rtac_base/include/rtac_base/</path>
    <filename>include_2rtac__base_2geometry_8h.html</filename>
    <namespace>rtac</namespace>
    <member kind="function">
      <type>Eigen::Matrix&lt; T, D, 1 &gt;</type>
      <name>find_noncolinear</name>
      <anchorfile>include_2rtac__base_2geometry_8h.html</anchorfile>
      <anchor>acabeee574816b5b2ed8cf6c525e7045b</anchor>
      <arglist>(const Eigen::Matrix&lt; T, D, 1 &gt; &amp;v)</arglist>
    </member>
    <member kind="function">
      <type>Eigen::Matrix&lt; T, D, 1 &gt;</type>
      <name>find_orthogonal</name>
      <anchorfile>include_2rtac__base_2geometry_8h.html</anchorfile>
      <anchor>a992debeb1ffc9c4c2942a41d7f7fd264</anchor>
      <arglist>(const Eigen::Matrix&lt; T, D, 1 &gt; &amp;v)</arglist>
    </member>
    <member kind="function">
      <type>Eigen::Matrix&lt; T, D, D &gt;</type>
      <name>orthonormalized</name>
      <anchorfile>include_2rtac__base_2geometry_8h.html</anchorfile>
      <anchor>a0e8e5ceddd287631aaaa79c18221d84b</anchor>
      <arglist>(const Eigen::Matrix&lt; T, D, D &gt; &amp;M, T tol=1e-6)</arglist>
    </member>
    <member kind="function">
      <type>Matrix3&lt; T &gt;</type>
      <name>look_at</name>
      <anchorfile>include_2rtac__base_2geometry_8h.html</anchorfile>
      <anchor>a1b6ad866725b405192d8eb3d83e2ff52</anchor>
      <arglist>(const Vector3&lt; T &gt; &amp;target, const Vector3&lt; T &gt; &amp;position, const Vector3&lt; T &gt; &amp;up)</arglist>
    </member>
  </compound>
  <compound kind="file">
    <name>files.h</name>
    <path>/home/pnarvor/work/narval/code/rtac_base/include/rtac_base/</path>
    <filename>files_8h.html</filename>
    <namespace>rtac</namespace>
    <member kind="function">
      <type>std::basic_istream&lt; CharT, Traits &gt; &amp;</type>
      <name>getline</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a2a256c6728424a025b3a8e3ffae11da9</anchor>
      <arglist>(std::basic_istream&lt; CharT, Traits &gt; &amp;input, std::basic_string&lt; CharT, Traits, Allocator &gt; &amp;str, CharT delim)</arglist>
    </member>
    <member kind="function">
      <type>std::basic_istream&lt; CharT, Traits &gt; &amp;</type>
      <name>getline</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a322bff136d3964994f18d2f6e45aafe8</anchor>
      <arglist>(std::basic_istream&lt; CharT, Traits &gt; &amp;&amp;input, std::basic_string&lt; CharT, Traits, Allocator &gt; &amp;str, CharT delim)</arglist>
    </member>
    <member kind="function">
      <type>std::basic_istream&lt; CharT, Traits &gt; &amp;</type>
      <name>getline</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>aa097913f82e94289f6e2c51061aa849e</anchor>
      <arglist>(std::basic_istream&lt; CharT, Traits &gt; &amp;input, std::basic_string&lt; CharT, Traits, Allocator &gt; &amp;str)</arglist>
    </member>
    <member kind="function">
      <type>std::basic_istream&lt; CharT, Traits &gt; &amp;</type>
      <name>getline</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a7e2451c11c7f07b1837af074e466abac</anchor>
      <arglist>(std::basic_istream&lt; CharT, Traits &gt; &amp;&amp;input, std::basic_string&lt; CharT, Traits, Allocator &gt; &amp;str)</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>rtac_data_path</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a327e6421660b10c5167f6d846e9d3c73</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>PathList</type>
      <name>rtac_data_paths</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a9193dc347e730a34fbf679b6341054b3</anchor>
      <arglist>(const std::string &amp;delimiter=&quot;:&quot;)</arglist>
    </member>
    <member kind="function">
      <type>PathList</type>
      <name>find</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>aee435138de5e590197bc8b3092275bed</anchor>
      <arglist>(const std::string &amp;reString=&quot;.*&quot;, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>PathList</type>
      <name>find</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a80104b26596b6b7696c29c9c1d63cc2c</anchor>
      <arglist>(const std::string &amp;reString, const char *path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>PathList</type>
      <name>find</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>add17ca63d0cfe65ac9c240440e547106</anchor>
      <arglist>(const std::string &amp;reString, const std::string &amp;path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>PathList</type>
      <name>find</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>adcf97909b89c6a224d0133564a5a5562</anchor>
      <arglist>(const std::string &amp;reString, const PathList &amp;path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>find_one</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a33d388bd20475f3e1e5e3b62cd9b7c1f</anchor>
      <arglist>(const std::string &amp;reString=&quot;.*&quot;, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>find_one</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a11e1b63aef5ec3819dc7b564157e6001</anchor>
      <arglist>(const std::string &amp;reString, const char *path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>find_one</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a47fa449ce547c16e219f0aabc261a3e9</anchor>
      <arglist>(const std::string &amp;reString, const std::string &amp;path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>find_one</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a849bd98e925e06f6e6a72608172b2e5e</anchor>
      <arglist>(const std::string &amp;reString, const PathList &amp;path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>write_pgm</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a56539f51dc3b181bb668b8f22fc53d5c</anchor>
      <arglist>(const std::string &amp;path, size_t width, size_t height, const char *data, const std::string &amp;comment=&quot;&quot;)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>write_ppm</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a94e5b8920d5aa41af06479e9fcc25ea7</anchor>
      <arglist>(const std::string &amp;path, size_t width, size_t height, const char *data, const std::string &amp;comment=&quot;&quot;)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>read_ppm</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a7de764274846d8c45b4cea8acc1f1908</anchor>
      <arglist>(const std::string &amp;path, size_t &amp;width, size_t &amp;height, std::vector&lt; uint8_t &gt; &amp;data)</arglist>
    </member>
  </compound>
  <compound kind="file">
    <name>files.cpp</name>
    <path>/home/pnarvor/work/narval/code/rtac_base/src/</path>
    <filename>files_8cpp.html</filename>
    <includes id="files_8h" name="files.h" local="no" imported="no">rtac_base/files.h</includes>
    <namespace>rtac</namespace>
    <member kind="function">
      <type>std::string</type>
      <name>rtac_data_path</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a327e6421660b10c5167f6d846e9d3c73</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>PathList</type>
      <name>rtac_data_paths</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a9193dc347e730a34fbf679b6341054b3</anchor>
      <arglist>(const std::string &amp;delimiter=&quot;:&quot;)</arglist>
    </member>
    <member kind="function">
      <type>PathList</type>
      <name>find</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>aee435138de5e590197bc8b3092275bed</anchor>
      <arglist>(const std::string &amp;reString=&quot;.*&quot;, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>PathList</type>
      <name>find</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a80104b26596b6b7696c29c9c1d63cc2c</anchor>
      <arglist>(const std::string &amp;reString, const char *path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>PathList</type>
      <name>find</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>add17ca63d0cfe65ac9c240440e547106</anchor>
      <arglist>(const std::string &amp;reString, const std::string &amp;path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>PathList</type>
      <name>find</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>adcf97909b89c6a224d0133564a5a5562</anchor>
      <arglist>(const std::string &amp;reString, const PathList &amp;path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>find_one</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a33d388bd20475f3e1e5e3b62cd9b7c1f</anchor>
      <arglist>(const std::string &amp;reString=&quot;.*&quot;, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>find_one</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a11e1b63aef5ec3819dc7b564157e6001</anchor>
      <arglist>(const std::string &amp;reString, const char *path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>find_one</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a47fa449ce547c16e219f0aabc261a3e9</anchor>
      <arglist>(const std::string &amp;reString, const std::string &amp;path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>find_one</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a849bd98e925e06f6e6a72608172b2e5e</anchor>
      <arglist>(const std::string &amp;reString, const PathList &amp;path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>write_pgm</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a56539f51dc3b181bb668b8f22fc53d5c</anchor>
      <arglist>(const std::string &amp;path, size_t width, size_t height, const char *data, const std::string &amp;comment=&quot;&quot;)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>write_ppm</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a94e5b8920d5aa41af06479e9fcc25ea7</anchor>
      <arglist>(const std::string &amp;path, size_t width, size_t height, const char *data, const std::string &amp;comment=&quot;&quot;)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>read_ppm</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a7de764274846d8c45b4cea8acc1f1908</anchor>
      <arglist>(const std::string &amp;path, size_t &amp;width, size_t &amp;height, std::vector&lt; uint8_t &gt; &amp;data)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>rtac::cuda::Addition</name>
    <filename>structrtac_1_1cuda_1_1Addition.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>rtac::ArrayDim</name>
    <filename>classrtac_1_1ArrayDim.html</filename>
    <templarg>VectorT</templarg>
    <base>DimExpression&lt; ArrayDim&lt; VectorT &gt; &gt;</base>
  </compound>
  <compound kind="struct">
    <name>rtac::Bounds</name>
    <filename>structrtac_1_1Bounds.html</filename>
    <templarg></templarg>
    <templarg>SizeV</templarg>
  </compound>
  <compound kind="class">
    <name>Bounds&lt; float &gt;</name>
    <filename>structrtac_1_1Bounds.html</filename>
  </compound>
  <compound kind="class">
    <name>Bounds&lt; float, 1 &gt;</name>
    <filename>structrtac_1_1Bounds.html</filename>
  </compound>
  <compound kind="struct">
    <name>rtac::Bounds&lt; T, 1 &gt;</name>
    <filename>structrtac_1_1Bounds_3_01T_00_011_01_4.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>rtac::BuildTarget</name>
    <filename>classrtac_1_1BuildTarget.html</filename>
    <class kind="struct">rtac::BuildTarget::CircularDependencyError</class>
  </compound>
  <compound kind="class">
    <name>rtac::BuildTargetHandle</name>
    <filename>classrtac_1_1BuildTargetHandle.html</filename>
  </compound>
  <compound kind="class">
    <name>rtac::CallbackQueue</name>
    <filename>classrtac_1_1CallbackQueue.html</filename>
    <templarg>ArgTypes</templarg>
  </compound>
  <compound kind="class">
    <name>rtac::external::Chunk</name>
    <filename>classrtac_1_1external_1_1Chunk.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>rtac::external::ChunkContainer</name>
    <filename>classrtac_1_1external_1_1ChunkContainer.html</filename>
    <templarg></templarg>
    <class kind="class">rtac::external::ChunkContainer::ConstIterator</class>
  </compound>
  <compound kind="struct">
    <name>rtac::BuildTarget::CircularDependencyError</name>
    <filename>structrtac_1_1BuildTarget_1_1CircularDependencyError.html</filename>
  </compound>
  <compound kind="class">
    <name>rtac::time::Clock</name>
    <filename>classrtac_1_1time_1_1Clock.html</filename>
    <member kind="function">
      <type>void</type>
      <name>reset</name>
      <anchorfile>classrtac_1_1time_1_1Clock.html</anchorfile>
      <anchor>a61ed43a8323d839b9694ead691c55d3b</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>T</type>
      <name>now</name>
      <anchorfile>classrtac_1_1time_1_1Clock.html</anchorfile>
      <anchor>a6a51b1533142547e321e77fe4b324bd1</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>T</type>
      <name>interval</name>
      <anchorfile>classrtac_1_1time_1_1Clock.html</anchorfile>
      <anchor>ad940754e8355374ab3bded18ae0916c6</anchor>
      <arglist>()</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rtac::Complex</name>
    <filename>classrtac_1_1Complex.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>rtac::external::ChunkContainer::ConstIterator</name>
    <filename>classrtac_1_1external_1_1ChunkContainer_1_1ConstIterator.html</filename>
  </compound>
  <compound kind="struct">
    <name>rtac::ConstVectorView</name>
    <filename>structrtac_1_1ConstVectorView.html</filename>
    <templarg></templarg>
    <base>rtac::VectorView&lt; const T &gt;</base>
  </compound>
  <compound kind="class">
    <name>rtac::cuda::DeviceObject</name>
    <filename>classrtac_1_1cuda_1_1DeviceObject.html</filename>
    <templarg></templarg>
    <base>rtac::cuda::DeviceObjectPtr</base>
  </compound>
  <compound kind="class">
    <name>rtac::cuda::DeviceObjectBase</name>
    <filename>classrtac_1_1cuda_1_1DeviceObjectBase.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>rtac::cuda::DeviceObjectPtr</name>
    <filename>classrtac_1_1cuda_1_1DeviceObjectPtr.html</filename>
    <templarg></templarg>
    <base>rtac::cuda::DeviceObjectBase</base>
  </compound>
  <compound kind="class">
    <name>rtac::cuda::DeviceVector</name>
    <filename>classrtac_1_1cuda_1_1DeviceVector.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::DimExpression</name>
    <filename>structrtac_1_1DimExpression.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>DimExpression&lt; ArrayDim&lt; VectorT &gt; &gt;</name>
    <filename>structrtac_1_1DimExpression.html</filename>
  </compound>
  <compound kind="class">
    <name>DimExpression&lt; LinearDim &gt;</name>
    <filename>structrtac_1_1DimExpression.html</filename>
  </compound>
  <compound kind="class">
    <name>rtac::DimIterator</name>
    <filename>classrtac_1_1DimIterator.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::cuda::Division</name>
    <filename>structrtac_1_1cuda_1_1Division.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::details::EigenFetchVectorElement</name>
    <filename>structrtac_1_1details_1_1EigenFetchVectorElement.html</filename>
    <templarg>Rows</templarg>
    <templarg>Cols</templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::details::EigenFetchVectorElement&lt; 1, Cols &gt;</name>
    <filename>structrtac_1_1details_1_1EigenFetchVectorElement_3_011_00_01Cols_01_4.html</filename>
    <templarg>Cols</templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::details::EigenFetchVectorElement&lt; Rows, 1 &gt;</name>
    <filename>structrtac_1_1details_1_1EigenFetchVectorElement_3_01Rows_00_011_01_4.html</filename>
    <templarg>Rows</templarg>
  </compound>
  <compound kind="class">
    <name>happly::Element</name>
    <filename>classhapply_1_1Element.html</filename>
    <member kind="function">
      <type></type>
      <name>Element</name>
      <anchorfile>classhapply_1_1Element.html</anchorfile>
      <anchor>af08fe995c4124e50f107556a7a654ded</anchor>
      <arglist>(const std::string &amp;name_, size_t count_)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>hasProperty</name>
      <anchorfile>classhapply_1_1Element.html</anchorfile>
      <anchor>a802a3c6fcff5ec462d92d545c6fa7048</anchor>
      <arglist>(const std::string &amp;target)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>hasPropertyType</name>
      <anchorfile>classhapply_1_1Element.html</anchorfile>
      <anchor>a947611fd0338f3646f5f6768bffcd3ff</anchor>
      <arglist>(const std::string &amp;target)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; std::string &gt;</type>
      <name>getPropertyNames</name>
      <anchorfile>classhapply_1_1Element.html</anchorfile>
      <anchor>af37017b42a9ae5e335557e41d606e8e5</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; Property &gt; &amp;</type>
      <name>getPropertyPtr</name>
      <anchorfile>classhapply_1_1Element.html</anchorfile>
      <anchor>a85d5c06999d73667e3323f28913c653e</anchor>
      <arglist>(const std::string &amp;target)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>addProperty</name>
      <anchorfile>classhapply_1_1Element.html</anchorfile>
      <anchor>a0bbbe9acfd2ca454e723d20b9ed54b9f</anchor>
      <arglist>(const std::string &amp;propertyName, const std::vector&lt; T &gt; &amp;data)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>addListProperty</name>
      <anchorfile>classhapply_1_1Element.html</anchorfile>
      <anchor>adac0222244e60b8ab55191c8e9d57d58</anchor>
      <arglist>(const std::string &amp;propertyName, const std::vector&lt; std::vector&lt; T &gt;&gt; &amp;data)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; T &gt;</type>
      <name>getProperty</name>
      <anchorfile>classhapply_1_1Element.html</anchorfile>
      <anchor>a52a5a0629b3bed4eec7aa497046944df</anchor>
      <arglist>(const std::string &amp;propertyName)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; T &gt;</type>
      <name>getPropertyType</name>
      <anchorfile>classhapply_1_1Element.html</anchorfile>
      <anchor>a977434df80605b76e580bbefc1628743</anchor>
      <arglist>(const std::string &amp;propertyName)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; std::vector&lt; T &gt; &gt;</type>
      <name>getListProperty</name>
      <anchorfile>classhapply_1_1Element.html</anchorfile>
      <anchor>a00f920df4e402ed9f5c84c9755d447de</anchor>
      <arglist>(const std::string &amp;propertyName)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; std::vector&lt; T &gt; &gt;</type>
      <name>getListPropertyType</name>
      <anchorfile>classhapply_1_1Element.html</anchorfile>
      <anchor>ae0c2d1b3616c57d0822c929874e7c7f8</anchor>
      <arglist>(const std::string &amp;propertyName)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; std::vector&lt; T &gt; &gt;</type>
      <name>getListPropertyAnySign</name>
      <anchorfile>classhapply_1_1Element.html</anchorfile>
      <anchor>ae50f5237daec63b3df29963b4702a7cf</anchor>
      <arglist>(const std::string &amp;propertyName)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>validate</name>
      <anchorfile>classhapply_1_1Element.html</anchorfile>
      <anchor>a925fc71912aca6bca5021df980071826</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>writeHeader</name>
      <anchorfile>classhapply_1_1Element.html</anchorfile>
      <anchor>acc0d3e0a3fbca24c53b838d8a6d69640</anchor>
      <arglist>(std::ostream &amp;outStream)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>writeDataASCII</name>
      <anchorfile>classhapply_1_1Element.html</anchorfile>
      <anchor>a994324a2638d2ee7284a86966324adc3</anchor>
      <arglist>(std::ostream &amp;outStream)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>writeDataBinary</name>
      <anchorfile>classhapply_1_1Element.html</anchorfile>
      <anchor>a8430ca772b86480a728305e0f10b5c01</anchor>
      <arglist>(std::ostream &amp;outStream)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>writeDataBinaryBigEndian</name>
      <anchorfile>classhapply_1_1Element.html</anchorfile>
      <anchor>a4c7a1ac57b53bf834cc9778b6f004342</anchor>
      <arglist>(std::ostream &amp;outStream)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; D &gt;</type>
      <name>getDataFromPropertyRecursive</name>
      <anchorfile>classhapply_1_1Element.html</anchorfile>
      <anchor>a5bd7edcb167fea88ade3e8a6a63e2d98</anchor>
      <arglist>(Property *prop)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; std::vector&lt; D &gt; &gt;</type>
      <name>getDataFromListPropertyRecursive</name>
      <anchorfile>classhapply_1_1Element.html</anchorfile>
      <anchor>aa055bfdc237d6ec5d302101fc04456f0</anchor>
      <arglist>(Property *prop)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rtac::time::FrameCounter</name>
    <filename>classrtac_1_1time_1_1FrameCounter.html</filename>
  </compound>
  <compound kind="struct">
    <name>rtac::Function1D</name>
    <filename>structrtac_1_1Function1D.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>Function1D&lt; LinearFunction1D &gt;</name>
    <filename>structrtac_1_1Function1D.html</filename>
  </compound>
  <compound kind="class">
    <name>Function1D&lt; TexCoordScaler &gt;</name>
    <filename>structrtac_1_1Function1D.html</filename>
  </compound>
  <compound kind="struct">
    <name>rtac::Function2D</name>
    <filename>structrtac_1_1Function2D.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>Function2D&lt; TextureFunction2D&lt; T, XScalerT, YScalerT &gt; &gt;</name>
    <filename>structrtac_1_1Function2D.html</filename>
  </compound>
  <compound kind="class">
    <name>rtac::display::GLVector</name>
    <filename>classrtac_1_1display_1_1GLVector.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>rtac::HostVector</name>
    <filename>classrtac_1_1HostVector.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>rtac::Image</name>
    <filename>classrtac_1_1Image.html</filename>
    <templarg></templarg>
    <templarg>ContainerT</templarg>
    <base>ImageExpression&lt; Image&lt; T, HostVector &gt; &gt;</base>
  </compound>
  <compound kind="class">
    <name>Image&lt; T, VectorT &gt;</name>
    <filename>classrtac_1_1Image.html</filename>
    <base>ImageExpression&lt; Image&lt; T, VectorT &gt; &gt;</base>
  </compound>
  <compound kind="class">
    <name>rtac::external::ImageCodec</name>
    <filename>classrtac_1_1external_1_1ImageCodec.html</filename>
    <class kind="struct">rtac::external::ImageCodec::ImageInfo</class>
  </compound>
  <compound kind="class">
    <name>rtac::external::ImageCodecBase</name>
    <filename>classrtac_1_1external_1_1ImageCodecBase.html</filename>
  </compound>
  <compound kind="struct">
    <name>rtac::ImageExpression</name>
    <filename>classrtac_1_1ImageExpression.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>ImageExpression&lt; Image&lt; T, HostVector &gt; &gt;</name>
    <filename>classrtac_1_1ImageExpression.html</filename>
  </compound>
  <compound kind="class">
    <name>ImageExpression&lt; Image&lt; T, VectorT &gt; &gt;</name>
    <filename>classrtac_1_1ImageExpression.html</filename>
  </compound>
  <compound kind="class">
    <name>ImageExpression&lt; ImageView&lt; const T &gt; &gt;</name>
    <filename>classrtac_1_1ImageExpression.html</filename>
  </compound>
  <compound kind="class">
    <name>ImageExpression&lt; ScaledImage&lt; T, WDimT, HDimT, VectorT &gt; &gt;</name>
    <filename>classrtac_1_1ImageExpression.html</filename>
  </compound>
  <compound kind="class">
    <name>ImageExpression&lt; ScaledImageView&lt; const T, WDimT, HDimT &gt; &gt;</name>
    <filename>classrtac_1_1ImageExpression.html</filename>
  </compound>
  <compound kind="class">
    <name>ImageExpression&lt; ScaledImageView&lt; T, WDimT, HDimT &gt; &gt;</name>
    <filename>classrtac_1_1ImageExpression.html</filename>
  </compound>
  <compound kind="struct">
    <name>rtac::external::ImageCodec::ImageInfo</name>
    <filename>structrtac_1_1external_1_1ImageCodec_1_1ImageInfo.html</filename>
  </compound>
  <compound kind="class">
    <name>rtac::ImageView</name>
    <filename>classrtac_1_1ImageView.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>rtac::ImageView&lt; const T &gt;</name>
    <filename>classrtac_1_1ImageView_3_01const_01T_01_4.html</filename>
    <templarg></templarg>
    <base>ImageExpression&lt; ImageView&lt; const T &gt; &gt;</base>
  </compound>
  <compound kind="class">
    <name>rtac::algorithm::Interpolator</name>
    <filename>classrtac_1_1algorithm_1_1Interpolator.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>rtac::algorithm::InterpolatorCubicSpline</name>
    <filename>classrtac_1_1algorithm_1_1InterpolatorCubicSpline.html</filename>
    <templarg></templarg>
    <base>rtac::algorithm::InterpolatorInterface</base>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>interpolate</name>
      <anchorfile>classrtac_1_1algorithm_1_1InterpolatorCubicSpline.html</anchorfile>
      <anchor>af462f3cd7857989f87e7038044154b18</anchor>
      <arglist>(VectorView&lt; const T &gt; x, VectorView&lt; T &gt; y) const</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rtac::algorithm::InterpolatorInterface</name>
    <filename>classrtac_1_1algorithm_1_1InterpolatorInterface.html</filename>
    <templarg></templarg>
    <member kind="function">
      <type>unsigned int</type>
      <name>size</name>
      <anchorfile>classrtac_1_1algorithm_1_1InterpolatorInterface.html</anchorfile>
      <anchor>ac7cfed5cf624aa859cc1669277ed467e</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>Xconst_iterator</type>
      <name>lower_bound</name>
      <anchorfile>classrtac_1_1algorithm_1_1InterpolatorInterface.html</anchorfile>
      <anchor>adbfb353d71fe34063da2b4d00738f4a4</anchor>
      <arglist>(T x) const</arglist>
    </member>
    <member kind="function">
      <type>Indexes</type>
      <name>lower_bound_indexes</name>
      <anchorfile>classrtac_1_1algorithm_1_1InterpolatorInterface.html</anchorfile>
      <anchor>a1df2ae7ca143af76b8260902b7256d9a</anchor>
      <arglist>(VectorView&lt; const T &gt; x) const</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>interpolate</name>
      <anchorfile>classrtac_1_1algorithm_1_1InterpolatorInterface.html</anchorfile>
      <anchor>a037e4663d5c84ae8306ab34822c3d2dc</anchor>
      <arglist>(VectorView&lt; const T &gt; x, VectorView&lt; T &gt; y) const =0</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rtac::algorithm::InterpolatorLinear</name>
    <filename>classrtac_1_1algorithm_1_1InterpolatorLinear.html</filename>
    <templarg></templarg>
    <base>rtac::algorithm::InterpolatorInterface</base>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>interpolate</name>
      <anchorfile>classrtac_1_1algorithm_1_1InterpolatorLinear.html</anchorfile>
      <anchor>a6231c2437403f3c021ad11248e930843</anchor>
      <arglist>(VectorView&lt; const T &gt; x, VectorView&lt; T &gt; y) const</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rtac::algorithm::InterpolatorNearest</name>
    <filename>classrtac_1_1algorithm_1_1InterpolatorNearest.html</filename>
    <templarg></templarg>
    <base>rtac::algorithm::InterpolatorInterface</base>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>interpolate</name>
      <anchorfile>classrtac_1_1algorithm_1_1InterpolatorNearest.html</anchorfile>
      <anchor>a7263a2eed8b2042abfe7cb69c590691c</anchor>
      <arglist>(VectorView&lt; const T &gt; x, VectorView&lt; T &gt; y) const</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>rtac::IsDimExpression</name>
    <filename>structrtac_1_1IsDimExpression.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::IsFunction1D</name>
    <filename>structrtac_1_1IsFunction1D.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::IsFunction2D</name>
    <filename>structrtac_1_1IsFunction2D.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::IsScaledImage</name>
    <filename>structrtac_1_1IsScaledImage.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>rtac::external::JPGCodec</name>
    <filename>classrtac_1_1external_1_1JPGCodec.html</filename>
    <base>rtac::external::ImageCodecBase</base>
  </compound>
  <compound kind="class">
    <name>rtac::LinearDim</name>
    <filename>classrtac_1_1LinearDim.html</filename>
    <base>DimExpression&lt; LinearDim &gt;</base>
  </compound>
  <compound kind="struct">
    <name>rtac::LinearFunction1D</name>
    <filename>structrtac_1_1LinearFunction1D.html</filename>
    <base>Function1D&lt; LinearFunction1D &gt;</base>
  </compound>
  <compound kind="struct">
    <name>rtac::cuda::linear::StridesType::LinearIndex</name>
    <filename>structrtac_1_1cuda_1_1linear_1_1StridesType_1_1LinearIndex.html</filename>
    <templarg>row</templarg>
    <templarg>col</templarg>
  </compound>
  <compound kind="class">
    <name>rtac::MappedPointer</name>
    <filename>classrtac_1_1MappedPointer.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>rtac::MappedPointer&lt; const VectorT &gt;</name>
    <filename>classrtac_1_1MappedPointer_3_01const_01VectorT_01_4.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>rtac::MappedPointerImpl</name>
    <filename>classrtac_1_1MappedPointerImpl.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>rtac::MappedPointerImpl&lt; const VectorT &gt;</name>
    <filename>classrtac_1_1MappedPointerImpl_3_01const_01VectorT_01_4.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>rtac::cuda::linear::MatrixBase</name>
    <filename>classrtac_1_1cuda_1_1linear_1_1MatrixBase.html</filename>
    <templarg></templarg>
    <templarg></templarg>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>MatrixBase&lt; T, S, MatrixData&lt; T, S &gt; &gt;</name>
    <filename>classrtac_1_1cuda_1_1linear_1_1MatrixBase.html</filename>
  </compound>
  <compound kind="class">
    <name>rtac::cuda::linear::MatrixData</name>
    <filename>classrtac_1_1cuda_1_1linear_1_1MatrixData.html</filename>
    <templarg></templarg>
    <templarg></templarg>
    <base>MatrixBase&lt; T, S, MatrixData&lt; T, S &gt; &gt;</base>
  </compound>
  <compound kind="class">
    <name>rtac::Mesh</name>
    <filename>classrtac_1_1Mesh.html</filename>
    <templarg></templarg>
    <templarg></templarg>
    <templarg></templarg>
    <templarg></templarg>
    <templarg>V</templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::external::MtlMaterial</name>
    <filename>structrtac_1_1external_1_1MtlMaterial.html</filename>
  </compound>
  <compound kind="struct">
    <name>rtac::cuda::Multiplication</name>
    <filename>structrtac_1_1cuda_1_1Multiplication.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::nmea::NmeaError</name>
    <filename>structrtac_1_1nmea_1_1NmeaError.html</filename>
  </compound>
  <compound kind="class">
    <name>rtac::external::ObjLoader</name>
    <filename>classrtac_1_1external_1_1ObjLoader.html</filename>
  </compound>
  <compound kind="class">
    <name>rtac::cuda::PinnedVector</name>
    <filename>classrtac_1_1cuda_1_1PinnedVector.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>happly::PLYData</name>
    <filename>classhapply_1_1PLYData.html</filename>
    <member kind="function">
      <type></type>
      <name>PLYData</name>
      <anchorfile>classhapply_1_1PLYData.html</anchorfile>
      <anchor>a1b6f8c52967e17424e1c4a9fd76be7e0</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>PLYData</name>
      <anchorfile>classhapply_1_1PLYData.html</anchorfile>
      <anchor>a0126613a424bfef4c09d1c38c47671ac</anchor>
      <arglist>(const std::string &amp;filename, bool verbose=false)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>PLYData</name>
      <anchorfile>classhapply_1_1PLYData.html</anchorfile>
      <anchor>aeb552f956e3342b9fdc1b3c8bf01a567</anchor>
      <arglist>(std::istream &amp;inStream, bool verbose=false)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>validate</name>
      <anchorfile>classhapply_1_1PLYData.html</anchorfile>
      <anchor>aacbcae9701db44d47ab0e51c2ddcbcaf</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>write</name>
      <anchorfile>classhapply_1_1PLYData.html</anchorfile>
      <anchor>a8c57f9aaa6277d222fdeb0a558f9072d</anchor>
      <arglist>(const std::string &amp;filename, DataFormat format=DataFormat::ASCII)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>write</name>
      <anchorfile>classhapply_1_1PLYData.html</anchorfile>
      <anchor>aa4d393de8c49f37c1e10d5cafa579c25</anchor>
      <arglist>(std::ostream &amp;outStream, DataFormat format=DataFormat::ASCII)</arglist>
    </member>
    <member kind="function">
      <type>Element &amp;</type>
      <name>getElement</name>
      <anchorfile>classhapply_1_1PLYData.html</anchorfile>
      <anchor>a68c97280bfdcbac5a1fe2b189c0f39f0</anchor>
      <arglist>(const std::string &amp;target)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>hasElement</name>
      <anchorfile>classhapply_1_1PLYData.html</anchorfile>
      <anchor>a3f7e6610d381c3756d72f1b21b8bb461</anchor>
      <arglist>(const std::string &amp;target)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; std::string &gt;</type>
      <name>getElementNames</name>
      <anchorfile>classhapply_1_1PLYData.html</anchorfile>
      <anchor>a06c9fdb863d6b6262b0989b497bb1f62</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>addElement</name>
      <anchorfile>classhapply_1_1PLYData.html</anchorfile>
      <anchor>a683e469e41c4c7707cee844b3ec68a41</anchor>
      <arglist>(const std::string &amp;name, size_t count)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; std::array&lt; double, 3 &gt; &gt;</type>
      <name>getVertexPositions</name>
      <anchorfile>classhapply_1_1PLYData.html</anchorfile>
      <anchor>a486c5daef175402f3a66afdbee85d50e</anchor>
      <arglist>(const std::string &amp;vertexElementName=&quot;vertex&quot;)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; std::array&lt; unsigned char, 3 &gt; &gt;</type>
      <name>getVertexColors</name>
      <anchorfile>classhapply_1_1PLYData.html</anchorfile>
      <anchor>ab323244a4cfe0ecafd08d5dc1a49fa0c</anchor>
      <arglist>(const std::string &amp;vertexElementName=&quot;vertex&quot;)</arglist>
    </member>
    <member kind="function">
      <type>std::vector&lt; std::vector&lt; T &gt; &gt;</type>
      <name>getFaceIndices</name>
      <anchorfile>classhapply_1_1PLYData.html</anchorfile>
      <anchor>a9f56108d0e1c9297785edf61e7694740</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>addVertexPositions</name>
      <anchorfile>classhapply_1_1PLYData.html</anchorfile>
      <anchor>ad85b62c377e9130af3642224c6b94a81</anchor>
      <arglist>(std::vector&lt; std::array&lt; double, 3 &gt;&gt; &amp;vertexPositions)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>addVertexColors</name>
      <anchorfile>classhapply_1_1PLYData.html</anchorfile>
      <anchor>a4e3d6a06141ea018af0b4930127cb5b2</anchor>
      <arglist>(std::vector&lt; std::array&lt; unsigned char, 3 &gt;&gt; &amp;colors)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>addVertexColors</name>
      <anchorfile>classhapply_1_1PLYData.html</anchorfile>
      <anchor>aef14557af75ed4eb2475dac142c07f59</anchor>
      <arglist>(std::vector&lt; std::array&lt; double, 3 &gt;&gt; &amp;colors)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>addFaceIndices</name>
      <anchorfile>classhapply_1_1PLYData.html</anchorfile>
      <anchor>a660e4d7dc7239add8b1e010beae5ea95</anchor>
      <arglist>(std::vector&lt; std::vector&lt; T &gt;&gt; &amp;indices)</arglist>
    </member>
    <member kind="variable">
      <type>std::vector&lt; std::string &gt;</type>
      <name>comments</name>
      <anchorfile>classhapply_1_1PLYData.html</anchorfile>
      <anchor>a99acf9794b2b33a87165884043678841</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::vector&lt; std::string &gt;</type>
      <name>objInfoComments</name>
      <anchorfile>classhapply_1_1PLYData.html</anchorfile>
      <anchor>acb6d06a2a9e99b8630f70969835855cc</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rtac::external::PNGCodec</name>
    <filename>classrtac_1_1external_1_1PNGCodec.html</filename>
    <base>rtac::external::ImageCodecBase</base>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>write_image</name>
      <anchorfile>classrtac_1_1external_1_1PNGCodec.html</anchorfile>
      <anchor>ac0bff8df57b991621168b0e5c708524c</anchor>
      <arglist>(const std::string &amp;path, const ImageCodec::ImageInfo &amp;info, const unsigned char *data, bool invertRows)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>rtac::external::PNGColorType</name>
    <filename>structrtac_1_1external_1_1PNGColorType.html</filename>
    <templarg>Channels</templarg>
    <templarg></templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::external::PNGColorType&lt; 2, T &gt;</name>
    <filename>structrtac_1_1external_1_1PNGColorType_3_012_00_01T_01_4.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::external::PNGColorType&lt; 3, T &gt;</name>
    <filename>structrtac_1_1external_1_1PNGColorType_3_013_00_01T_01_4.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::external::PNGColorType&lt; 4, T &gt;</name>
    <filename>structrtac_1_1external_1_1PNGColorType_3_014_00_01T_01_4.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::external::PNGPixelError</name>
    <filename>structrtac_1_1external_1_1PNGPixelError.html</filename>
    <templarg>D</templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::external::PNGPixelType</name>
    <filename>structrtac_1_1external_1_1PNGPixelType.html</filename>
    <templarg>ChannelCount</templarg>
    <templarg>ChannelDepth</templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::external::PNGScalar</name>
    <filename>structrtac_1_1external_1_1PNGScalar.html</filename>
    <templarg>Depth</templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::external::PNGScalar&lt; 16 &gt;</name>
    <filename>structrtac_1_1external_1_1PNGScalar_3_0116_01_4.html</filename>
  </compound>
  <compound kind="struct">
    <name>rtac::external::PNGScalar&lt; 8 &gt;</name>
    <filename>structrtac_1_1external_1_1PNGScalar_3_018_01_4.html</filename>
  </compound>
  <compound kind="struct">
    <name>rtac::PODWrapper</name>
    <filename>structrtac_1_1PODWrapper.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::Point2</name>
    <filename>structrtac_1_1Point2.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::Point3</name>
    <filename>structrtac_1_1Point3.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>Point3&lt; float &gt;</name>
    <filename>structrtac_1_1Point3.html</filename>
  </compound>
  <compound kind="struct">
    <name>rtac::Point4</name>
    <filename>structrtac_1_1Point4.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>rtac::PointCloud</name>
    <filename>classrtac_1_1PointCloud.html</filename>
    <templarg></templarg>
    <member kind="function">
      <type>void</type>
      <name>resize</name>
      <anchorfile>classrtac_1_1PointCloud.html</anchorfile>
      <anchor>ac00a308bf0366f9ac79869f89b9589de</anchor>
      <arglist>(size_t n)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>resize</name>
      <anchorfile>classrtac_1_1PointCloud.html</anchorfile>
      <anchor>af8b7c05b05ad62b8e75a3c74f40483d0</anchor>
      <arglist>(uint32_t width, uint32_t height)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>push_back</name>
      <anchorfile>classrtac_1_1PointCloud.html</anchorfile>
      <anchor>aae4f62efccd614fb94fe70ab37cb9bbb</anchor>
      <arglist>(const PointType &amp;p)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>operator const PointCloudT &amp;</name>
      <anchorfile>classrtac_1_1PointCloud.html</anchorfile>
      <anchor>af39f71ee715d964c80e941a8dd1dda7d</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>operator PointCloudT &amp;</name>
      <anchorfile>classrtac_1_1PointCloud.html</anchorfile>
      <anchor>ae735386396ea8d119a90caaf8023d8a1</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>const_iterator</type>
      <name>begin</name>
      <anchorfile>classrtac_1_1PointCloud.html</anchorfile>
      <anchor>a6acd3618866c9ee5c4d2bd232d8cf88d</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>const_iterator</type>
      <name>end</name>
      <anchorfile>classrtac_1_1PointCloud.html</anchorfile>
      <anchor>a4e0b8883da0ab33c38d71782d953c8e0</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>empty</name>
      <anchorfile>classrtac_1_1PointCloud.html</anchorfile>
      <anchor>a84e9375dd0265a02db9056219ebbd55f</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>export_ply</name>
      <anchorfile>classrtac_1_1PointCloud.html</anchorfile>
      <anchor>af36426a2b5a5e631e53e681f1b9bd5cf</anchor>
      <arglist>(const std::string &amp;path, bool ascii=false) const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>export_ply</name>
      <anchorfile>classrtac_1_1PointCloud.html</anchorfile>
      <anchor>a0e47df95d59c6743025ac061fc3d49ef</anchor>
      <arglist>(std::ostream &amp;os, bool ascii=false) const</arglist>
    </member>
    <member kind="function">
      <type>happly::PLYData</type>
      <name>export_ply</name>
      <anchorfile>classrtac_1_1PointCloud.html</anchorfile>
      <anchor>a593cba03cfde6a5c172fef378a686457</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static PointCloud&lt; PointCloudT &gt;</type>
      <name>from_ply</name>
      <anchorfile>classrtac_1_1PointCloud.html</anchorfile>
      <anchor>a9a6baa2c38593b54e8c010594604d945</anchor>
      <arglist>(const std::string &amp;path)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static PointCloud&lt; PointCloudT &gt;</type>
      <name>from_ply</name>
      <anchorfile>classrtac_1_1PointCloud.html</anchorfile>
      <anchor>aa401e6c4754266ea338f64ebae6fd1c1</anchor>
      <arglist>(std::istream &amp;is)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static PointCloud&lt; PointCloudT &gt;</type>
      <name>from_ply</name>
      <anchorfile>classrtac_1_1PointCloud.html</anchorfile>
      <anchor>a79f05081f2cde907ca0e312d109bc3ee</anchor>
      <arglist>(happly::PLYData &amp;data)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rtac::PointCloudBase</name>
    <filename>classrtac_1_1PointCloudBase.html</filename>
    <templarg></templarg>
    <member kind="function">
      <type>void</type>
      <name>resize</name>
      <anchorfile>classrtac_1_1PointCloudBase.html</anchorfile>
      <anchor>aeb87a5f921faf68ac985c10ae38de741</anchor>
      <arglist>(size_t n)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>push_back</name>
      <anchorfile>classrtac_1_1PointCloudBase.html</anchorfile>
      <anchor>a38506f925fb60369c58b2005ae2516e5</anchor>
      <arglist>(const PointT &amp;p)</arglist>
    </member>
    <member kind="function">
      <type>const_iterator</type>
      <name>begin</name>
      <anchorfile>classrtac_1_1PointCloudBase.html</anchorfile>
      <anchor>ad2c5a639549685a419d0391f9f034c6d</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>const_iterator</type>
      <name>end</name>
      <anchorfile>classrtac_1_1PointCloudBase.html</anchorfile>
      <anchor>ae7fcf93792c86b3ae0fb8a0da9755772</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>empty</name>
      <anchorfile>classrtac_1_1PointCloudBase.html</anchorfile>
      <anchor>afaaed5c84db518f098346269f045fc89</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="variable">
      <type>uint32_t</type>
      <name>height</name>
      <anchorfile>classrtac_1_1PointCloudBase.html</anchorfile>
      <anchor>a06daf3a37e5e14f0440197656caafa18</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>Vector4&lt; float &gt;</type>
      <name>sensor_origin_</name>
      <anchorfile>classrtac_1_1PointCloudBase.html</anchorfile>
      <anchor>a4ead72b3cc1c1e630583be704f79ce51</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>rtac::Pose</name>
    <filename>structrtac_1_1Pose.html</filename>
    <templarg></templarg>
    <member kind="function">
      <type>RTAC_HOSTDEVICE Pose &amp;</type>
      <name>look_at</name>
      <anchorfile>structrtac_1_1Pose.html</anchorfile>
      <anchor>aa172963393b29d9b82cc299d05170cb5</anchor>
      <arglist>(const Vec3 &amp;target, const Vec3 &amp;position, const Vec3 &amp;up=Vec3({0, 0, 1}))</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>happly::Property</name>
    <filename>classhapply_1_1Property.html</filename>
    <member kind="function">
      <type></type>
      <name>Property</name>
      <anchorfile>classhapply_1_1Property.html</anchorfile>
      <anchor>a6c7588117a819f8e40d23bc01889bf59</anchor>
      <arglist>(const std::string &amp;name_)</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>reserve</name>
      <anchorfile>classhapply_1_1Property.html</anchorfile>
      <anchor>ae0c5c11f4de1714d6d5a92e98bb13d0a</anchor>
      <arglist>(size_t capacity)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>parseNext</name>
      <anchorfile>classhapply_1_1Property.html</anchorfile>
      <anchor>a2c4563aebd62d85ecf1eb41608ebecf2</anchor>
      <arglist>(const std::vector&lt; std::string &gt; &amp;tokens, size_t &amp;currEntry)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>readNext</name>
      <anchorfile>classhapply_1_1Property.html</anchorfile>
      <anchor>a4137231485778feb691b57bb726b76c2</anchor>
      <arglist>(std::istream &amp;stream)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>readNextBigEndian</name>
      <anchorfile>classhapply_1_1Property.html</anchorfile>
      <anchor>ad7b81e34a4b920ed664bc1a351b7f438</anchor>
      <arglist>(std::istream &amp;stream)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>writeHeader</name>
      <anchorfile>classhapply_1_1Property.html</anchorfile>
      <anchor>addfcf4d092cacd686fdf748b5a677aba</anchor>
      <arglist>(std::ostream &amp;outStream)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>writeDataASCII</name>
      <anchorfile>classhapply_1_1Property.html</anchorfile>
      <anchor>a852bf52201d2e456b04e7b16327d24df</anchor>
      <arglist>(std::ostream &amp;outStream, size_t iElement)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>writeDataBinary</name>
      <anchorfile>classhapply_1_1Property.html</anchorfile>
      <anchor>ab16e76ba43d6eaeeab67b3a8e2ceb089</anchor>
      <arglist>(std::ostream &amp;outStream, size_t iElement)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>writeDataBinaryBigEndian</name>
      <anchorfile>classhapply_1_1Property.html</anchorfile>
      <anchor>ae2fa81866da608f39d366631aa109987</anchor>
      <arglist>(std::ostream &amp;outStream, size_t iElement)=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual size_t</type>
      <name>size</name>
      <anchorfile>classhapply_1_1Property.html</anchorfile>
      <anchor>ab10eae59b91dacd82e26b0833a4c9f5e</anchor>
      <arglist>()=0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual std::string</type>
      <name>propertyTypeName</name>
      <anchorfile>classhapply_1_1Property.html</anchorfile>
      <anchor>a51a8c2bf37df9d13f975e5f9ef185b09</anchor>
      <arglist>()=0</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>rtac::cuda::linear::ProxyLoader</name>
    <filename>structrtac_1_1cuda_1_1linear_1_1ProxyLoader.html</filename>
    <templarg></templarg>
    <templarg></templarg>
    <templarg></templarg>
    <templarg>N</templarg>
  </compound>
  <compound kind="class">
    <name>rtac::Rectangle</name>
    <filename>classrtac_1_1Rectangle.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::cuda::Ref</name>
    <filename>structrtac_1_1cuda_1_1Ref.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>rtac::ScaledImage</name>
    <filename>classrtac_1_1ScaledImage.html</filename>
    <templarg></templarg>
    <templarg></templarg>
    <templarg></templarg>
    <templarg>VectorT</templarg>
    <base>ScaledImageExpression&lt; ScaledImage&lt; T, WDimT, HDimT, VectorT &gt; &gt;</base>
  </compound>
  <compound kind="struct">
    <name>rtac::ScaledImageExpression</name>
    <filename>structrtac_1_1ScaledImageExpression.html</filename>
    <templarg></templarg>
    <base>rtac::ImageExpression</base>
  </compound>
  <compound kind="class">
    <name>ScaledImageExpression&lt; ScaledImage&lt; T, WDimT, HDimT, VectorT &gt; &gt;</name>
    <filename>structrtac_1_1ScaledImageExpression.html</filename>
    <base>ImageExpression&lt; ScaledImage&lt; T, WDimT, HDimT, VectorT &gt; &gt;</base>
  </compound>
  <compound kind="class">
    <name>ScaledImageExpression&lt; ScaledImageView&lt; const T, WDimT, HDimT &gt; &gt;</name>
    <filename>structrtac_1_1ScaledImageExpression.html</filename>
    <base>ImageExpression&lt; ScaledImageView&lt; const T, WDimT, HDimT &gt; &gt;</base>
  </compound>
  <compound kind="class">
    <name>ScaledImageExpression&lt; ScaledImageView&lt; T, WDimT, HDimT &gt; &gt;</name>
    <filename>structrtac_1_1ScaledImageExpression.html</filename>
    <base>ImageExpression&lt; ScaledImageView&lt; T, WDimT, HDimT &gt; &gt;</base>
  </compound>
  <compound kind="class">
    <name>rtac::ScaledImageView</name>
    <filename>classrtac_1_1ScaledImageView.html</filename>
    <templarg></templarg>
    <templarg></templarg>
    <templarg></templarg>
    <base>ScaledImageExpression&lt; ScaledImageView&lt; T, WDimT, HDimT &gt; &gt;</base>
  </compound>
  <compound kind="class">
    <name>rtac::ScaledImageView&lt; const T, WDimT, HDimT &gt;</name>
    <filename>classrtac_1_1ScaledImageView_3_01const_01T_00_01WDimT_00_01HDimT_01_4.html</filename>
    <templarg></templarg>
    <templarg></templarg>
    <templarg></templarg>
    <base>ScaledImageExpression&lt; ScaledImageView&lt; const T, WDimT, HDimT &gt; &gt;</base>
  </compound>
  <compound kind="struct">
    <name>rtac::Shape</name>
    <filename>structrtac_1_1Shape.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>Shape&lt; uint32_t &gt;</name>
    <filename>structrtac_1_1Shape.html</filename>
  </compound>
  <compound kind="struct">
    <name>rtac::signal::SincFunction</name>
    <filename>structrtac_1_1signal_1_1SincFunction.html</filename>
    <templarg></templarg>
    <member kind="function">
      <type>T</type>
      <name>physical_span</name>
      <anchorfile>structrtac_1_1signal_1_1SincFunction.html</anchorfile>
      <anchor>a05f44de42ffd4d6506412227b5e67c6e</anchor>
      <arglist>(T resolution) const</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rtac::signal::SinFunction</name>
    <filename>classrtac_1_1signal_1_1SinFunction.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>rtac::SonarPing2D</name>
    <filename>classrtac_1_1SonarPing2D.html</filename>
    <templarg></templarg>
    <templarg>VectorT</templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::cuda::linear::StridesType</name>
    <filename>structrtac_1_1cuda_1_1linear_1_1StridesType.html</filename>
    <templarg>R</templarg>
    <templarg>C</templarg>
    <templarg>Rstride</templarg>
    <templarg>Cstride</templarg>
    <class kind="struct">rtac::cuda::linear::StridesType::LinearIndex</class>
  </compound>
  <compound kind="struct">
    <name>rtac::cuda::Substraction</name>
    <filename>structrtac_1_1cuda_1_1Substraction.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::cuda::TexCoordScaler</name>
    <filename>structrtac_1_1cuda_1_1TexCoordScaler.html</filename>
    <base>Function1D&lt; TexCoordScaler &gt;</base>
  </compound>
  <compound kind="class">
    <name>rtac::cuda::Texture2D</name>
    <filename>classrtac_1_1cuda_1_1Texture2D.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::cuda::TextureFunction2D</name>
    <filename>structrtac_1_1cuda_1_1TextureFunction2D.html</filename>
    <templarg></templarg>
    <templarg></templarg>
    <templarg></templarg>
    <base>Function2D&lt; TextureFunction2D&lt; T, XScalerT, YScalerT &gt; &gt;</base>
  </compound>
  <compound kind="struct">
    <name>rtac::cuda::TextureView1D</name>
    <filename>structrtac_1_1cuda_1_1TextureView1D.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::cuda::TextureView2D</name>
    <filename>structrtac_1_1cuda_1_1TextureView2D.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>happly::TypedListProperty</name>
    <filename>classhapply_1_1TypedListProperty.html</filename>
    <templarg></templarg>
    <base>happly::Property</base>
    <member kind="function">
      <type></type>
      <name>TypedListProperty</name>
      <anchorfile>classhapply_1_1TypedListProperty.html</anchorfile>
      <anchor>ad908207551f8db827cace459aa292ebe</anchor>
      <arglist>(const std::string &amp;name_, int listCountBytes_)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>TypedListProperty</name>
      <anchorfile>classhapply_1_1TypedListProperty.html</anchorfile>
      <anchor>aec6254320db55aa697f0c21127432e93</anchor>
      <arglist>(const std::string &amp;name_, const std::vector&lt; std::vector&lt; T &gt;&gt; &amp;data_)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>reserve</name>
      <anchorfile>classhapply_1_1TypedListProperty.html</anchorfile>
      <anchor>ad47105a8d951b729d63d9fc980718ca1</anchor>
      <arglist>(size_t capacity) override</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>parseNext</name>
      <anchorfile>classhapply_1_1TypedListProperty.html</anchorfile>
      <anchor>a82987c6d859c755bbee7a3f750218a3d</anchor>
      <arglist>(const std::vector&lt; std::string &gt; &amp;tokens, size_t &amp;currEntry) override</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>readNext</name>
      <anchorfile>classhapply_1_1TypedListProperty.html</anchorfile>
      <anchor>adc2a22aefe6ded2d26747a6e9d3a041d</anchor>
      <arglist>(std::istream &amp;stream) override</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>readNextBigEndian</name>
      <anchorfile>classhapply_1_1TypedListProperty.html</anchorfile>
      <anchor>a8a0759b2d2e19ae114e4a1869c94f41e</anchor>
      <arglist>(std::istream &amp;stream) override</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>writeHeader</name>
      <anchorfile>classhapply_1_1TypedListProperty.html</anchorfile>
      <anchor>af39a19bd15040d8ddcaf438dae5026d0</anchor>
      <arglist>(std::ostream &amp;outStream) override</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>writeDataASCII</name>
      <anchorfile>classhapply_1_1TypedListProperty.html</anchorfile>
      <anchor>adab8d4abac40526324b899498c38edc4</anchor>
      <arglist>(std::ostream &amp;outStream, size_t iElement) override</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>writeDataBinary</name>
      <anchorfile>classhapply_1_1TypedListProperty.html</anchorfile>
      <anchor>ab00c774ee6fb4af66f917803aee40749</anchor>
      <arglist>(std::ostream &amp;outStream, size_t iElement) override</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>writeDataBinaryBigEndian</name>
      <anchorfile>classhapply_1_1TypedListProperty.html</anchorfile>
      <anchor>a5e79291523d5f334d4f9f1cf2c0c3ac4</anchor>
      <arglist>(std::ostream &amp;outStream, size_t iElement) override</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual size_t</type>
      <name>size</name>
      <anchorfile>classhapply_1_1TypedListProperty.html</anchorfile>
      <anchor>a8235eb563141ea1db958773634dcd5a6</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual std::string</type>
      <name>propertyTypeName</name>
      <anchorfile>classhapply_1_1TypedListProperty.html</anchorfile>
      <anchor>a399f4b50fe90524b7c23b6bff6a436f4</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="variable">
      <type>std::vector&lt; T &gt;</type>
      <name>flattenedData</name>
      <anchorfile>classhapply_1_1TypedListProperty.html</anchorfile>
      <anchor>a85e27412d73c602e8da57cf196263bab</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::vector&lt; size_t &gt;</type>
      <name>flattenedIndexStart</name>
      <anchorfile>classhapply_1_1TypedListProperty.html</anchorfile>
      <anchor>a597e85896702aa5d0abe96e5ed55f6bc</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>listCountBytes</name>
      <anchorfile>classhapply_1_1TypedListProperty.html</anchorfile>
      <anchor>abc88b8b4ea004f765b7adabf3ee5b8f6</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>happly::TypedProperty</name>
    <filename>classhapply_1_1TypedProperty.html</filename>
    <templarg></templarg>
    <base>happly::Property</base>
    <member kind="function">
      <type></type>
      <name>TypedProperty</name>
      <anchorfile>classhapply_1_1TypedProperty.html</anchorfile>
      <anchor>a681ebc36bee7415c3ce2464ec91e896e</anchor>
      <arglist>(const std::string &amp;name_)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>TypedProperty</name>
      <anchorfile>classhapply_1_1TypedProperty.html</anchorfile>
      <anchor>a0dce483afa4f66fd21f2fe992110360d</anchor>
      <arglist>(const std::string &amp;name_, const std::vector&lt; T &gt; &amp;data_)</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>reserve</name>
      <anchorfile>classhapply_1_1TypedProperty.html</anchorfile>
      <anchor>a424ff8712ef15877980dc8c5b3d859be</anchor>
      <arglist>(size_t capacity) override</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>parseNext</name>
      <anchorfile>classhapply_1_1TypedProperty.html</anchorfile>
      <anchor>add4f214d89d6545d36ac7c86f59a791f</anchor>
      <arglist>(const std::vector&lt; std::string &gt; &amp;tokens, size_t &amp;currEntry) override</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>readNext</name>
      <anchorfile>classhapply_1_1TypedProperty.html</anchorfile>
      <anchor>a4263e537653696550008459eb9c2851d</anchor>
      <arglist>(std::istream &amp;stream) override</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>readNextBigEndian</name>
      <anchorfile>classhapply_1_1TypedProperty.html</anchorfile>
      <anchor>a64e0acfdc21b9aa62c4c567ce45c4a8b</anchor>
      <arglist>(std::istream &amp;stream) override</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>writeHeader</name>
      <anchorfile>classhapply_1_1TypedProperty.html</anchorfile>
      <anchor>a86840b25bc0bf81e0baae704b6c036e4</anchor>
      <arglist>(std::ostream &amp;outStream) override</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>writeDataASCII</name>
      <anchorfile>classhapply_1_1TypedProperty.html</anchorfile>
      <anchor>a4d1f8e688ad86db260d14fa56adc104d</anchor>
      <arglist>(std::ostream &amp;outStream, size_t iElement) override</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>writeDataBinary</name>
      <anchorfile>classhapply_1_1TypedProperty.html</anchorfile>
      <anchor>a76bd09b9976b926ef86905881c6a06b4</anchor>
      <arglist>(std::ostream &amp;outStream, size_t iElement) override</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual void</type>
      <name>writeDataBinaryBigEndian</name>
      <anchorfile>classhapply_1_1TypedProperty.html</anchorfile>
      <anchor>a90e4b626881b8957e3dbd4bed3543eeb</anchor>
      <arglist>(std::ostream &amp;outStream, size_t iElement) override</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual size_t</type>
      <name>size</name>
      <anchorfile>classhapply_1_1TypedProperty.html</anchorfile>
      <anchor>a11da9c21748533fca6022b22e363f1bb</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="function" virtualness="virtual">
      <type>virtual std::string</type>
      <name>propertyTypeName</name>
      <anchorfile>classhapply_1_1TypedProperty.html</anchorfile>
      <anchor>a84d7d499c5b7b986d4243304e617e3a3</anchor>
      <arglist>() override</arglist>
    </member>
    <member kind="variable">
      <type>std::vector&lt; T &gt;</type>
      <name>data</name>
      <anchorfile>classhapply_1_1TypedProperty.html</anchorfile>
      <anchor>ac8f58492c4a650edd5643a1f79f43d3c</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rtac::VectorView</name>
    <filename>classrtac_1_1VectorView.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="class">
    <name>rtac::VectorView&lt; const T &gt;</name>
    <filename>classrtac_1_1VectorView_3_01const_01T_01_4.html</filename>
    <templarg></templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::external::VertexId</name>
    <filename>structrtac_1_1external_1_1VertexId.html</filename>
  </compound>
  <compound kind="namespace">
    <name>rtac</name>
    <filename>namespacertac.html</filename>
    <class kind="class">rtac::ArrayDim</class>
    <class kind="struct">rtac::Bounds</class>
    <class kind="struct">rtac::Bounds&lt; T, 1 &gt;</class>
    <class kind="class">rtac::BuildTarget</class>
    <class kind="class">rtac::BuildTargetHandle</class>
    <class kind="class">rtac::CallbackQueue</class>
    <class kind="class">rtac::Complex</class>
    <class kind="struct">rtac::ConstVectorView</class>
    <class kind="struct">rtac::DimExpression</class>
    <class kind="class">rtac::DimIterator</class>
    <class kind="struct">rtac::Function1D</class>
    <class kind="struct">rtac::Function2D</class>
    <class kind="class">rtac::HostVector</class>
    <class kind="class">rtac::Image</class>
    <class kind="struct">rtac::ImageExpression</class>
    <class kind="class">rtac::ImageView</class>
    <class kind="class">rtac::ImageView&lt; const T &gt;</class>
    <class kind="struct">rtac::IsDimExpression</class>
    <class kind="struct">rtac::IsFunction1D</class>
    <class kind="struct">rtac::IsFunction2D</class>
    <class kind="struct">rtac::IsScaledImage</class>
    <class kind="class">rtac::LinearDim</class>
    <class kind="struct">rtac::LinearFunction1D</class>
    <class kind="class">rtac::MappedPointer</class>
    <class kind="class">rtac::MappedPointer&lt; const VectorT &gt;</class>
    <class kind="class">rtac::MappedPointerImpl</class>
    <class kind="class">rtac::MappedPointerImpl&lt; const VectorT &gt;</class>
    <class kind="class">rtac::Mesh</class>
    <class kind="struct">rtac::PODWrapper</class>
    <class kind="struct">rtac::Point2</class>
    <class kind="struct">rtac::Point3</class>
    <class kind="struct">rtac::Point4</class>
    <class kind="class">rtac::PointCloud</class>
    <class kind="class">rtac::PointCloudBase</class>
    <class kind="struct">rtac::Pose</class>
    <class kind="class">rtac::Rectangle</class>
    <class kind="class">rtac::ScaledImage</class>
    <class kind="struct">rtac::ScaledImageExpression</class>
    <class kind="class">rtac::ScaledImageView</class>
    <class kind="class">rtac::ScaledImageView&lt; const T, WDimT, HDimT &gt;</class>
    <class kind="struct">rtac::Shape</class>
    <class kind="class">rtac::SonarPing2D</class>
    <class kind="class">rtac::VectorView</class>
    <class kind="class">rtac::VectorView&lt; const T &gt;</class>
    <member kind="function">
      <type>auto</type>
      <name>make_view</name>
      <anchorfile>namespacertac.html</anchorfile>
      <anchor>a2ab36a9817f772a7701bf9abd0ad2b00</anchor>
      <arglist>(const Container &amp;container)</arglist>
    </member>
    <member kind="function">
      <type>auto</type>
      <name>make_view</name>
      <anchorfile>namespacertac.html</anchorfile>
      <anchor>a70b0b698cf78e8319c3b92ff505e3993</anchor>
      <arglist>(const std::vector&lt; T &gt; &amp;container)</arglist>
    </member>
  </compound>
  <compound kind="page">
    <name>md_README</name>
    <title>rtac_base</title>
    <filename>md_README</filename>
  </compound>
</tagfile>
