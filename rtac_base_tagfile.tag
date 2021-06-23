<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile doxygen_version="1.9.2" doxygen_gitid="9d2b2e839ee5a7c6d7a99784d138460aea9324fd">
  <compound kind="file">
    <name>files.h</name>
    <path>/home/pnarvor/work/rtac/code/rtac_base/include/rtac_base/</path>
    <filename>files_8h.html</filename>
    <member kind="function">
      <type>std::string</type>
      <name>rtac_data_path</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a0a67dbbc3c19e8bf4b2d58cbd8acca4f</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>PathList</type>
      <name>rtac_data_paths</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a4c24f2dbaf98c62d63e9b04d3774532b</anchor>
      <arglist>(const std::string &amp;delimiter=&quot;:&quot;)</arglist>
    </member>
    <member kind="function">
      <type>PathList</type>
      <name>find</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a3ff1d89f3b5678abf05238d7870b722c</anchor>
      <arglist>(const std::string &amp;reString=&quot;.*&quot;, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>PathList</type>
      <name>find</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a466ace97b2fabe12f060b00afc3b9e9b</anchor>
      <arglist>(const std::string &amp;reString, const char *path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>PathList</type>
      <name>find</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>ac058cfb364ddd4f5587b37f0cce3c3b9</anchor>
      <arglist>(const std::string &amp;reString, const std::string &amp;path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>PathList</type>
      <name>find</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a1e9910713dcb717849175cde76bd4f46</anchor>
      <arglist>(const std::string &amp;reString, const PathList &amp;path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>find_one</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a2ad3293d382aece807d7f8bbab4999bb</anchor>
      <arglist>(const std::string &amp;reString=&quot;.*&quot;, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>find_one</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>afcc9b0f5dcb569adcface9e6e55b900e</anchor>
      <arglist>(const std::string &amp;reString, const char *path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>find_one</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a0a5e2ad218ca2704861e550d5f2398bb</anchor>
      <arglist>(const std::string &amp;reString, const std::string &amp;path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>find_one</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>af207b412a6dd21d1a1620c4ce654f4e0</anchor>
      <arglist>(const std::string &amp;reString, const PathList &amp;path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>write_pgm</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a0c233dbf680ffc968a5f5029285dc46c</anchor>
      <arglist>(const std::string &amp;path, size_t width, size_t height, const char *data, const std::string &amp;comment=&quot;&quot;)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>write_ppm</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>afbc1697c36ae7e10d0bfc9e9988eea28</anchor>
      <arglist>(const std::string &amp;path, size_t width, size_t height, const char *data, const std::string &amp;comment=&quot;&quot;)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>read_ppm</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a89eeeb7507ab06b9613739f3d44700f6</anchor>
      <arglist>(const std::string &amp;path, size_t &amp;width, size_t &amp;height, std::vector&lt; uint8_t &gt; &amp;data)</arglist>
    </member>
  </compound>
  <compound kind="file">
    <name>geometry.h</name>
    <path>/home/pnarvor/work/rtac/code/rtac_base/include/rtac_base/</path>
    <filename>geometry_8h.html</filename>
    <member kind="function">
      <type>Eigen::Matrix&lt; T, D, 1 &gt;</type>
      <name>find_noncolinear</name>
      <anchorfile>geometry_8h.html</anchorfile>
      <anchor>afef90abcb03b37f492b04dd758b36141</anchor>
      <arglist>(const Eigen::Matrix&lt; T, D, 1 &gt; &amp;v)</arglist>
    </member>
    <member kind="function">
      <type>Eigen::Matrix&lt; T, D, 1 &gt;</type>
      <name>find_orthogonal</name>
      <anchorfile>geometry_8h.html</anchorfile>
      <anchor>af12fd24bfa9d879030007ac8595c8b64</anchor>
      <arglist>(const Eigen::Matrix&lt; T, D, 1 &gt; &amp;v)</arglist>
    </member>
    <member kind="function">
      <type>Eigen::Matrix&lt; T, D, D &gt;</type>
      <name>orthonormalized</name>
      <anchorfile>geometry_8h.html</anchorfile>
      <anchor>aab7011801858ff3f225a2826fab09a91</anchor>
      <arglist>(const Eigen::Matrix&lt; T, D, D &gt; &amp;M, T tol=1e-6)</arglist>
    </member>
    <member kind="function">
      <type>Matrix3&lt; T &gt;</type>
      <name>look_at</name>
      <anchorfile>geometry_8h.html</anchorfile>
      <anchor>aee7f254bafc93884d14503b456dafc11</anchor>
      <arglist>(const Vector3&lt; T &gt; &amp;target, const Vector3&lt; T &gt; &amp;position, const Vector3&lt; T &gt; &amp;up)</arglist>
    </member>
  </compound>
  <compound kind="file">
    <name>files.cpp</name>
    <path>/home/pnarvor/work/rtac/code/rtac_base/src/</path>
    <filename>files_8cpp.html</filename>
    <includes id="files_8h" name="files.h" local="no" imported="no">rtac_base/files.h</includes>
    <member kind="function">
      <type>std::string</type>
      <name>rtac_data_path</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a0a67dbbc3c19e8bf4b2d58cbd8acca4f</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>PathList</type>
      <name>rtac_data_paths</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a4c24f2dbaf98c62d63e9b04d3774532b</anchor>
      <arglist>(const std::string &amp;delimiter=&quot;:&quot;)</arglist>
    </member>
    <member kind="function">
      <type>PathList</type>
      <name>find</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a3ff1d89f3b5678abf05238d7870b722c</anchor>
      <arglist>(const std::string &amp;reString=&quot;.*&quot;, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>PathList</type>
      <name>find</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a466ace97b2fabe12f060b00afc3b9e9b</anchor>
      <arglist>(const std::string &amp;reString, const char *path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>PathList</type>
      <name>find</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>ac058cfb364ddd4f5587b37f0cce3c3b9</anchor>
      <arglist>(const std::string &amp;reString, const std::string &amp;path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>PathList</type>
      <name>find</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a1e9910713dcb717849175cde76bd4f46</anchor>
      <arglist>(const std::string &amp;reString, const PathList &amp;path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>find_one</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a2ad3293d382aece807d7f8bbab4999bb</anchor>
      <arglist>(const std::string &amp;reString=&quot;.*&quot;, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>find_one</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>afcc9b0f5dcb569adcface9e6e55b900e</anchor>
      <arglist>(const std::string &amp;reString, const char *path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>find_one</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a0a5e2ad218ca2704861e550d5f2398bb</anchor>
      <arglist>(const std::string &amp;reString, const std::string &amp;path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>find_one</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>af207b412a6dd21d1a1620c4ce654f4e0</anchor>
      <arglist>(const std::string &amp;reString, const PathList &amp;path, bool followSimlink=true)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>write_pgm</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a0c233dbf680ffc968a5f5029285dc46c</anchor>
      <arglist>(const std::string &amp;path, size_t width, size_t height, const char *data, const std::string &amp;comment=&quot;&quot;)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>write_ppm</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>afbc1697c36ae7e10d0bfc9e9988eea28</anchor>
      <arglist>(const std::string &amp;path, size_t width, size_t height, const char *data, const std::string &amp;comment=&quot;&quot;)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>read_ppm</name>
      <anchorfile>files_8h.html</anchorfile>
      <anchor>a89eeeb7507ab06b9613739f3d44700f6</anchor>
      <arglist>(const std::string &amp;path, size_t &amp;width, size_t &amp;height, std::vector&lt; uint8_t &gt; &amp;data)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>rtac::cuda::Addition</name>
    <filename>structrtac_1_1cuda_1_1Addition.html</filename>
    <templarg>typename T</templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::types::Buildable</name>
    <filename>structrtac_1_1types_1_1Buildable.html</filename>
  </compound>
  <compound kind="class">
    <name>rtac::types::BuildableHandle</name>
    <filename>classrtac_1_1types_1_1BuildableHandle.html</filename>
    <templarg>typename BuildableT</templarg>
    <templarg>template&lt; typename T &gt; class PointerT</templarg>
  </compound>
  <compound kind="class">
    <name>rtac::types::BuildTarget</name>
    <filename>classrtac_1_1types_1_1BuildTarget.html</filename>
    <class kind="struct">rtac::types::BuildTarget::CircularDependencyError</class>
  </compound>
  <compound kind="class">
    <name>rtac::types::BuildTargetHandle</name>
    <filename>classrtac_1_1types_1_1BuildTargetHandle.html</filename>
    <templarg>typename TargetT</templarg>
    <templarg>template&lt; typename T &gt; class PointerT</templarg>
    <class kind="struct">rtac::types::BuildTargetHandle::Hash</class>
  </compound>
  <compound kind="class">
    <name>rtac::types::CallbackQueue</name>
    <filename>classrtac_1_1types_1_1CallbackQueue.html</filename>
    <templarg>class ... ArgTypes</templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::types::BuildTarget::CircularDependencyError</name>
    <filename>structrtac_1_1types_1_1BuildTarget_1_1CircularDependencyError.html</filename>
  </compound>
  <compound kind="class">
    <name>rtac::time::Clock</name>
    <filename>classrtac_1_1time_1_1Clock.html</filename>
    <member kind="function">
      <type>void</type>
      <name>reset</name>
      <anchorfile>classrtac_1_1time_1_1Clock.html</anchorfile>
      <anchor>a577974eceb129d314ef1965ff2a298e2</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>T</type>
      <name>now</name>
      <anchorfile>classrtac_1_1time_1_1Clock.html</anchorfile>
      <anchor>a6493d61b2970f4fa9f6e479f45bcd2f0</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>T</type>
      <name>interval</name>
      <anchorfile>classrtac_1_1time_1_1Clock.html</anchorfile>
      <anchor>ab1040a4f7f07c161254d3a7a823f8f2d</anchor>
      <arglist>()</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rtac::cuda::DeviceMesh</name>
    <filename>classrtac_1_1cuda_1_1DeviceMesh.html</filename>
    <templarg>typename PointT</templarg>
    <templarg>typename FaceT</templarg>
    <base>Mesh&lt; rtac::types::Point3&lt; float &gt;, rtac::types::Point3&lt; uint32_t &gt;, DeviceVector &gt;</base>
  </compound>
  <compound kind="class">
    <name>rtac::cuda::DeviceObject</name>
    <filename>classrtac_1_1cuda_1_1DeviceObject.html</filename>
    <templarg>class T</templarg>
    <base>rtac::cuda::DeviceObjectPtr</base>
  </compound>
  <compound kind="class">
    <name>rtac::cuda::DeviceObjectBase</name>
    <filename>classrtac_1_1cuda_1_1DeviceObjectBase.html</filename>
    <templarg>class T</templarg>
  </compound>
  <compound kind="class">
    <name>rtac::cuda::DeviceObjectPtr</name>
    <filename>classrtac_1_1cuda_1_1DeviceObjectPtr.html</filename>
    <templarg>class T</templarg>
    <base>rtac::cuda::DeviceObjectBase</base>
  </compound>
  <compound kind="class">
    <name>rtac::cuda::DeviceVector</name>
    <filename>classrtac_1_1cuda_1_1DeviceVector.html</filename>
    <templarg>typename T</templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::cuda::Division</name>
    <filename>structrtac_1_1cuda_1_1Division.html</filename>
    <templarg>typename T</templarg>
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
  <compound kind="struct">
    <name>rtac::types::ensure_shared_vector</name>
    <filename>structrtac_1_1types_1_1ensure__shared__vector.html</filename>
    <templarg>typename T</templarg>
  </compound>
  <compound kind="class">
    <name>rtac::time::FrameCounter</name>
    <filename>classrtac_1_1time_1_1FrameCounter.html</filename>
  </compound>
  <compound kind="struct">
    <name>rtac::types::BuildTargetHandle::Hash</name>
    <filename>structrtac_1_1types_1_1BuildTargetHandle_1_1Hash.html</filename>
  </compound>
  <compound kind="class">
    <name>rtac::cuda::HostVector</name>
    <filename>classrtac_1_1cuda_1_1HostVector.html</filename>
    <templarg>typename T</templarg>
  </compound>
  <compound kind="class">
    <name>rtac::types::MappedPointer</name>
    <filename>classrtac_1_1types_1_1MappedPointer.html</filename>
    <templarg>typename MappedT</templarg>
    <templarg>typename PointerT</templarg>
  </compound>
  <compound kind="class">
    <name>rtac::types::Mesh</name>
    <filename>classrtac_1_1types_1_1Mesh.html</filename>
    <templarg>typename PointT</templarg>
    <templarg>typename FaceT</templarg>
    <templarg>template&lt; typename &gt; class VectorT</templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::cuda::Multiplication</name>
    <filename>structrtac_1_1cuda_1_1Multiplication.html</filename>
    <templarg>typename T</templarg>
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
  <compound kind="struct">
    <name>rtac::types::Point2</name>
    <filename>structrtac_1_1types_1_1Point2.html</filename>
    <templarg>typename T</templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::types::Point3</name>
    <filename>structrtac_1_1types_1_1Point3.html</filename>
    <templarg>typename T</templarg>
  </compound>
  <compound kind="class">
    <name>rtac::types::PointCloud</name>
    <filename>classrtac_1_1types_1_1PointCloud.html</filename>
    <templarg>typename PointCloudT</templarg>
    <member kind="function">
      <type>void</type>
      <name>resize</name>
      <anchorfile>classrtac_1_1types_1_1PointCloud.html</anchorfile>
      <anchor>a36ddebc223956a068cae889c0f75c037</anchor>
      <arglist>(size_t n)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>resize</name>
      <anchorfile>classrtac_1_1types_1_1PointCloud.html</anchorfile>
      <anchor>a44c2ad30b6428e861412f7c32ce61880</anchor>
      <arglist>(uint32_t width, uint32_t height)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>push_back</name>
      <anchorfile>classrtac_1_1types_1_1PointCloud.html</anchorfile>
      <anchor>a4a030760c5fd46b46b99273488e33ad2</anchor>
      <arglist>(const PointType &amp;p)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>operator const PointCloudT &amp;</name>
      <anchorfile>classrtac_1_1types_1_1PointCloud.html</anchorfile>
      <anchor>a3face98899bcec17f6a2dee9ebd14653</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>operator PointCloudT &amp;</name>
      <anchorfile>classrtac_1_1types_1_1PointCloud.html</anchorfile>
      <anchor>a13e3e1fbca307626404e75a191b61b17</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>const_iterator</type>
      <name>begin</name>
      <anchorfile>classrtac_1_1types_1_1PointCloud.html</anchorfile>
      <anchor>a966ec9f74cd0325bff7df1f173ff4c4d</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>iterator</type>
      <name>begin</name>
      <anchorfile>classrtac_1_1types_1_1PointCloud.html</anchorfile>
      <anchor>a9b6b335d715e1de0e327b8fb8d44c672</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>const_iterator</type>
      <name>end</name>
      <anchorfile>classrtac_1_1types_1_1PointCloud.html</anchorfile>
      <anchor>a86e6bbfe122fd167855f9b67a0a157fc</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>iterator</type>
      <name>end</name>
      <anchorfile>classrtac_1_1types_1_1PointCloud.html</anchorfile>
      <anchor>ae415f4dcc77910f00f6cd2f3373283ab</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>empty</name>
      <anchorfile>classrtac_1_1types_1_1PointCloud.html</anchorfile>
      <anchor>a753ed43953e1a1eac79736c9c8d25177</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>export_ply</name>
      <anchorfile>classrtac_1_1types_1_1PointCloud.html</anchorfile>
      <anchor>a5f5bdb26c42cd7b732a3b43c8059575c</anchor>
      <arglist>(const std::string &amp;path, bool ascii=false) const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>export_ply</name>
      <anchorfile>classrtac_1_1types_1_1PointCloud.html</anchorfile>
      <anchor>a9ce2bd0b9376adb012ee4d73b3be039d</anchor>
      <arglist>(std::ostream &amp;os, bool ascii=false) const</arglist>
    </member>
    <member kind="function">
      <type>happly::PLYData</type>
      <name>export_ply</name>
      <anchorfile>classrtac_1_1types_1_1PointCloud.html</anchorfile>
      <anchor>aeab95c0e50fa58004414c16cf6ef21e7</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static PointCloud&lt; PointCloudT &gt;</type>
      <name>from_ply</name>
      <anchorfile>classrtac_1_1types_1_1PointCloud.html</anchorfile>
      <anchor>af5f05c87d830e5e0a10360d83ae0f944</anchor>
      <arglist>(const std::string &amp;path)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static PointCloud&lt; PointCloudT &gt;</type>
      <name>from_ply</name>
      <anchorfile>classrtac_1_1types_1_1PointCloud.html</anchorfile>
      <anchor>a2ae8d3965b8474b093d11c545cbcc9da</anchor>
      <arglist>(std::istream &amp;is)</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static PointCloud&lt; PointCloudT &gt;</type>
      <name>from_ply</name>
      <anchorfile>classrtac_1_1types_1_1PointCloud.html</anchorfile>
      <anchor>a4d11bb7f2a52f52ea2d9bbd92ab981ed</anchor>
      <arglist>(happly::PLYData &amp;data)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rtac::types::PointCloudBase</name>
    <filename>classrtac_1_1types_1_1PointCloudBase.html</filename>
    <templarg>typename PointT</templarg>
    <member kind="function">
      <type>void</type>
      <name>resize</name>
      <anchorfile>classrtac_1_1types_1_1PointCloudBase.html</anchorfile>
      <anchor>a8fd10ffb9e71486657e2394fee56b423</anchor>
      <arglist>(size_t n)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>push_back</name>
      <anchorfile>classrtac_1_1types_1_1PointCloudBase.html</anchorfile>
      <anchor>a9d1dfb982255bee7db950d69807948c9</anchor>
      <arglist>(const PointT &amp;p)</arglist>
    </member>
    <member kind="function">
      <type>const_iterator</type>
      <name>begin</name>
      <anchorfile>classrtac_1_1types_1_1PointCloudBase.html</anchorfile>
      <anchor>af38898df40f5309575f178c4d255d023</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>iterator</type>
      <name>begin</name>
      <anchorfile>classrtac_1_1types_1_1PointCloudBase.html</anchorfile>
      <anchor>a2f7e52a25503ed1d803dfcb39381fe81</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>const_iterator</type>
      <name>end</name>
      <anchorfile>classrtac_1_1types_1_1PointCloudBase.html</anchorfile>
      <anchor>a845d123b59b5623fcf07f719a95547fa</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>iterator</type>
      <name>end</name>
      <anchorfile>classrtac_1_1types_1_1PointCloudBase.html</anchorfile>
      <anchor>a9d072b0d85c9796d70df900f1afdd687</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>empty</name>
      <anchorfile>classrtac_1_1types_1_1PointCloudBase.html</anchorfile>
      <anchor>a2e8befd6e30a087bef64d9d1f73f4c4d</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="variable">
      <type>uint32_t</type>
      <name>height</name>
      <anchorfile>classrtac_1_1types_1_1PointCloudBase.html</anchorfile>
      <anchor>a30f1b2bd2940cd3aa43fda2eddeb82dd</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>Vector4&lt; float &gt;</type>
      <name>sensor_origin_</name>
      <anchorfile>classrtac_1_1types_1_1PointCloudBase.html</anchorfile>
      <anchor>acbbceb9d02d7d6ed45848162464349cd</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rtac::types::Pose</name>
    <filename>classrtac_1_1types_1_1Pose.html</filename>
    <templarg>typename T</templarg>
    <member kind="function">
      <type>Pose&lt; T &gt; &amp;</type>
      <name>operator*=</name>
      <anchorfile>classrtac_1_1types_1_1Pose.html</anchorfile>
      <anchor>a3e09377bc7bea6aeca17c5739d48acb2</anchor>
      <arglist>(const Pose&lt; T &gt; &amp;rhs)</arglist>
    </member>
    <member kind="function">
      <type>Pose&lt; T &gt; &amp;</type>
      <name>look_at</name>
      <anchorfile>classrtac_1_1types_1_1Pose.html</anchorfile>
      <anchor>ae265cfb0afd45433ac76b779c94deac3</anchor>
      <arglist>(const Vector3&lt; T &gt; &amp;target, const Vector3&lt; T &gt; &amp;position, const Vector3&lt; T &gt; &amp;up=Vector3&lt; T &gt;({0, 0, 1}))</arglist>
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
  <compound kind="class">
    <name>rtac::types::Rectangle</name>
    <filename>classrtac_1_1types_1_1Rectangle.html</filename>
    <templarg>typename T</templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::types::Shape</name>
    <filename>structrtac_1_1types_1_1Shape.html</filename>
    <templarg>typename T</templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::types::shared_vector_test</name>
    <filename>structrtac_1_1types_1_1shared__vector__test.html</filename>
    <templarg>typename OtherT</templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::types::shared_vector_test&lt; SharedVectorBase&lt; T &gt; &gt;</name>
    <filename>structrtac_1_1types_1_1shared__vector__test_3_01SharedVectorBase_3_01T_01_4_01_4.html</filename>
    <templarg>typename T</templarg>
  </compound>
  <compound kind="class">
    <name>rtac::types::SharedVectorBase</name>
    <filename>classrtac_1_1types_1_1SharedVectorBase.html</filename>
    <templarg>typename VectorT</templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::cuda::Substraction</name>
    <filename>structrtac_1_1cuda_1_1Substraction.html</filename>
    <templarg>typename T</templarg>
  </compound>
  <compound kind="class">
    <name>rtac::cuda::Texture2D</name>
    <filename>classrtac_1_1cuda_1_1Texture2D.html</filename>
    <templarg>typename T</templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::types::TupleTypeIndex</name>
    <filename>structrtac_1_1types_1_1TupleTypeIndex.html</filename>
    <templarg>class T</templarg>
    <templarg>class Tuple</templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::types::TupleTypeIndex&lt; T, std::tuple&lt; T, Types... &gt; &gt;</name>
    <filename>structrtac_1_1types_1_1TupleTypeIndex_3_01T_00_01std_1_1tuple_3_01T_00_01Types_8_8_8_01_4_01_4.html</filename>
    <templarg>class T</templarg>
    <templarg>class... Types</templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::types::TupleTypeIndex&lt; T, std::tuple&lt; U, Types... &gt; &gt;</name>
    <filename>structrtac_1_1types_1_1TupleTypeIndex_3_01T_00_01std_1_1tuple_3_01U_00_01Types_8_8_8_01_4_01_4.html</filename>
    <templarg>class T</templarg>
    <templarg>class U</templarg>
    <templarg>class... Types</templarg>
  </compound>
  <compound kind="struct">
    <name>rtac::types::TupleTypeIndex&lt; T, std::tuple&lt;&gt; &gt;</name>
    <filename>structrtac_1_1types_1_1TupleTypeIndex_3_01T_00_01std_1_1tuple_3_4_01_4.html</filename>
    <templarg>class T</templarg>
  </compound>
  <compound kind="class">
    <name>happly::TypedListProperty</name>
    <filename>classhapply_1_1TypedListProperty.html</filename>
    <templarg>class T</templarg>
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
    <templarg>class T</templarg>
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
  <compound kind="struct">
    <name>rtac::types::TypeInTuple</name>
    <filename>structrtac_1_1types_1_1TypeInTuple.html</filename>
    <templarg>class T</templarg>
    <templarg>class Tuple</templarg>
  </compound>
</tagfile>
